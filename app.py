import os
import jwt
import torch
import sqlparse
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- Configuraciones ---

# --- Configuraciones ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://sqlineage.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
SECRET_KEY = os.environ.get('SECRET_KEY', 'supersecretkey')

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Modelos de base de datos ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Historial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(150), nullable=False)
    fecha = db.Column(db.Date, nullable=False)
    tables = db.Column(db.JSON, nullable=False)
    columns = db.Column(db.JSON, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

# --- Modelo NLP ---
model_path = 'Codesql/SqlCodebert'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.to(device)
model.eval()

model.config.id2label = {
    0: "O", 1: "B-TRGTAB", 2: "I-TRGTAB", 3: "B-SRCTAB", 4: "I-SRCTAB",
    5: "B-TRGCOL", 6: "I-TRGCOL", 7: "B-SRCOL", 8: "I-SRCOL"
}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

# --- Token decorator ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token requerido'}), 401
        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user = decoded
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token inválido'}), 401
        return f(*args, **kwargs)
    return decorated

# --- Utilidades NLP ---
def dividir_consultas(sql_code):
    parsed = sqlparse.parse(sql_code)
    return [str(stmt).strip() for stmt in parsed if stmt.token_first(skip_cm=True)]

def tag_sql_query(query):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [model.config.id2label[p.item()] for p in preds]
    return [(tokenizer.convert_tokens_to_string([token]).strip(), label) for token, label in zip(tokens, labels) if token not in tokenizer.all_special_tokens]

def reconstruir_entidades(tagged):
    entidades, actual, tipo_actual = [], [], None
    def guardar_entidad(tokens, tipo):
        if not tokens: return None, None
        texto = "".join(tokens).replace(" ", "").strip()
        return texto, tipo

    for token, etiqueta in tagged + [(None, 'O')]:
        if etiqueta == 'O':
            if actual:
                texto, tipo = guardar_entidad(actual, tipo_actual)
                if texto: entidades.append((texto, tipo))
                actual, tipo_actual = [], None
            continue

        prefijo, tipo = etiqueta.split('-', 1)
        if prefijo == 'B' or tipo != tipo_actual:
            if actual:
                texto, tipo_ant = guardar_entidad(actual, tipo_actual)
                if texto: entidades.append((texto, tipo_ant))
            actual, tipo_actual = [token], tipo
        else:
            actual.append(token)

    return entidades

def extraer_select_segmento(query):
    query_upper = query.upper()
    select_idx, from_idx = query_upper.find('SELECT'), query_upper.find('FROM')
    return query[select_idx+6:from_idx].strip() if select_idx != -1 and from_idx != -1 else ""

def organizar_linaje(consultas):
    linaje = []
    for consulta in consultas:
        tagged = tag_sql_query(consulta)
        entidades = reconstruir_entidades(tagged)
        select_segment = extraer_select_segmento(consulta)
        columnas_select = [c.strip().split(" ")[0].split('.')[-1] for c in select_segment.split(',') if c.strip()] if select_segment else []
        src_tabs, tgt_tabs, src_cols, tgt_cols = [], [], [], []

        for valor, tipo in entidades:
            if tipo == 'SRCTAB': src_tabs.append(valor)
            elif tipo == 'TRGTAB': tgt_tabs.append(valor)
            elif tipo == 'SRCOL':
                if not columnas_select or any(c.lower() in valor.lower() for c in columnas_select):
                    src_cols.append(valor)
            elif tipo == 'TRGCOL': tgt_cols.append(valor)

        if 'INSERT' in consulta.upper():
            linaje.append({
                'source_tables': src_tabs,
                'source_columns': src_cols or ['Todas las columnas'],
                'target_table': tgt_tabs[0] if tgt_tabs else '',
                'target_columns': tgt_cols or src_cols
            })
        elif 'SELECT' in consulta.upper() and 'INTO' in consulta.upper():
            linaje.append({
                'source_tables': src_tabs,
                'source_columns': ['Todas las columnas'],
                'target_table': tgt_tabs[0] if tgt_tabs else '',
                'target_columns': ['Todas las columnas']
            })
        elif 'JOIN' in consulta.upper():
            for src in src_tabs:
                linaje.append({
                    'source_tables': [src],
                    'source_columns': src_cols or ['Todas las columnas'],
                    'target_table': tgt_tabs[0] if tgt_tabs else '',
                    'target_columns': tgt_cols or ['Todas las columnas']
                })

    return linaje

# --- Rutas API ---
@app.route('/api/tag_sql', methods=['PUT'])
@token_required
def tag_sql():
    sql_code = request.json.get('query', '')
    consultas = dividir_consultas(sql_code)
    resultado = organizar_linaje(consultas)
    return jsonify({"mensaje": "Linaje generado", "resultado": {"linaje": resultado}})

@app.route('/api/get_sql', methods=['GET'])
def get_sql():
    return jsonify({"mensaje": "No hay linaje disponible"}), 404

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username', '')
    email = data.get('email', '')
    password = data.get('password', '')

    if not all([username, email, password]):
        missing = [f for f in ["username", "email", "password"] if not data.get(f)]
        return jsonify({"message": "Faltan campos", "missing": missing}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "El email ya está registrado"}), 409

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "El nombre de usuario ya está registrado"}), 409

    hashed_pwd = generate_password_hash(password)
    user = User(username=username, email=email, password=hashed_pwd)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Usuario registrado correctamente"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email, password = data.get('email'), data.get('password')
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"message": "Correo invalido"}), 404

    if not check_password_hash(user.password, password):
        return jsonify({"message": "Contraseña incorrecta"}), 401

    expiration_time = datetime.utcnow() + timedelta(minutes=30)
    payload = {
        "user_id": user.id,
        "username": user.username,
        "exp": expiration_time
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return jsonify({"message": "Login exitoso", "token": token}), 200

@app.route('/api/historial', methods=['POST'])
@token_required
def guardar_historial():
    data = request.get_json() or {}
    nombre, linaje, user_id = data.get("nombre"), data.get("linaje"), data.get("user_id")
    if not all([nombre, linaje, user_id]):
        return jsonify({"message": "Nombre, linaje y usuario son requeridos"}), 400

    nuevo_historial = Historial(nombre=nombre, fecha=datetime.now(), tables=linaje, columns=linaje, user_id=user_id)
    db.session.add(nuevo_historial)
    db.session.commit()
    return jsonify({"message": "Linaje guardado exitosamente"}), 201

@app.route('/api/historial', methods=['GET'])
@token_required
def listar_historial():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'message': 'ID de usuario requerido'}), 400

    historiales = Historial.query.filter_by(user_id=user_id).all()
    return jsonify([{ 'id': h.id, 'nombre': h.nombre, 'fecha': h.fecha.strftime('%Y-%m-%d'), 'linaje': h.tables } for h in historiales]), 200

@app.route('/api/historial/<int:id>', methods=['DELETE'])
@token_required
def eliminar_historial(id):
    historial = Historial.query.get(id)
    if not historial:
        return jsonify({'message': 'Historial no encontrado'}), 404

    db.session.delete(historial)
    db.session.commit()
    return jsonify({'message': 'Historial eliminado correctamente'}), 200

@app.route('/api/historial/<int:id>', methods=['PUT'])
@token_required
def editar_historial(id):
    historial = Historial.query.get(id)
    if not historial:
        return jsonify({'message': 'Historial no encontrado'}), 404

    nuevo_nombre = request.get_json().get('nombre')
    if not nuevo_nombre:
        return jsonify({'message': 'El nombre es requerido'}), 400

    historial.nombre = nuevo_nombre
    db.session.commit()
    return jsonify({'message': 'Nombre del historial actualizado correctamente'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
