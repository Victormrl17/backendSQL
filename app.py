from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
CORS(app)

#--------------------------------------------------------------------

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://dblinejage_user:W3rcpNuH2qaaajWrYNMs7SbKAqy8hkIB@dpg-csua69ggph6c7388lc0g-a.ohio-postgres.render.com/dblinejage')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


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
    source_tables = db.Column(db.JSON, nullable=True)
    destination_table = db.Column(db.String(150), nullable=True)


with app.app_context():
    db.create_all()

#--------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained('./trained_model')
model = AutoModelForTokenClassification.from_pretrained('./trained_model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
response = ''

id_to_label = {
    0: "O",
    1: "B-TABLE",
    2: "I-TABLE",
    3: "B-COLUMN",
    4: "I-COLUMN"
}

# Función para etiquetar una consulta SQL
def tag_sql_query(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [id_to_label[id.item()] for id in predictions[0]]

    filtered_tokens_labels = [(token, label) for token, label in zip(tokens, labels) if token not in tokenizer.all_special_tokens]

    for i in range(len(filtered_tokens_labels) - 1):
        if filtered_tokens_labels[i][0].lower() == 'in' and filtered_tokens_labels[i + 1][0].lower() == 'to':
            if i + 2 < len(filtered_tokens_labels):
                filtered_tokens_labels[i + 2] = (filtered_tokens_labels[i + 2][0], 'B-TABLE')

    return filtered_tokens_labels

# Función para extraer tablas y columnas etiquetadas
def extract_tables_columns(tagged_query):
    tables = set()
    columns = set()
    source_tables = set()
    destination_table = None

    in_insert = False
    in_from = False

    for token, label in tagged_query:
        if token.lower() == "into":
            in_insert = True
        elif token.lower() == "from":
            in_from = True
        elif label == "B-TABLE" or label == "I-TABLE":
            if in_insert and destination_table is None:
                destination_table = token
            elif in_from:
                source_tables.add(token)
                tables.add(token)
            else:
                tables.add(token)
        elif label == "B-COLUMN" or label == "I-COLUMN":
            columns.add(token)
        if token.lower() == 'to' and label == 'O':
            next_index = tagged_query.index((token, label)) + 1
            if next_index < len(tagged_query):
                next_token, next_label = tagged_query[next_index]
                if next_label == 'B-TABLE':
                    destination_table = next_token
        if token == ";":
            in_insert = False
            in_from = False
    if not source_tables and len(tables) == 1:
        source_tables = tables.copy()

    return list(tables), list(columns), list(source_tables), destination_table

#--------------------------------------------------------------------

@app.route('/api/tag_sql', methods=['PUT'])
def tag_sql():
    global response
    data = request.json
    query = data.get('query', '')

    tagged_query = tag_sql_query(query)
    response = {
        "tagged_query": [{"token": token, "label": label} for token, label in tagged_query],
        "tables": [],
        "columns": [],
        "source_tables": [],
        "destination_table": None
    }


    tables, columns, source_tables, destination_table = extract_tables_columns(tagged_query)
    response["tables"] = tables
    response["columns"] = columns
    response["source_tables"] = source_tables
    response["destination_table"] = destination_table
    print(response)
    return jsonify({"mensaje": "Consulta guardada y procesada con éxito"})

@app.route('/api/get_sql', methods=['GET'])
def get_sql():
    if response:
        return jsonify(response)
    else:
        return jsonify({"mensaje": "No hay consultas procesadas disponibles"}), 404


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'message': 'Faltan campos obligatorios'}), 400

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, email=email, password=hashed_password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'Usuario registrado exitosamente'}), 201
    except:
        return jsonify({'message': 'Error al registrar usuario. El correo o nombre de usuario ya existe'}), 400


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, password):
        return jsonify({'message': 'Inicio de sesión exitoso', 'token': 'fake-jwt-token'}), 200
    else:
        return jsonify({'message': 'Correo o contraseña incorrectos'}), 401

@app.route('/api/historial', methods=['POST'])
def guardar_historial():
    data = request.get_json()
    nuevo_registro = Historial(
        nombre=data['nombre'],
        fecha=data['fecha'],
        tables=data['result']['tables'],
        columns=data['result']['columns'],
        source_tables=data['result'].get('sourceTables', []),
        destination_table=data['result'].get('destinationTable', None)
    )
    db.session.add(nuevo_registro)
    db.session.commit()
    return jsonify({'message': 'Registro guardado exitosamente'}), 201

@app.route('/api/historial', methods=['GET'])
def obtener_historial():
    historial = Historial.query.all()
    response = [{'id': h.id, 'nombre': h.nombre, 'fecha': h.fecha, 'tables': h.tables, 'columns': h.columns, 'source_tables': h.source_tables, 'destination_table': h.destination_table} for h in historial]
    return jsonify(response)

@app.route('/api/historial/<int:id>', methods=['DELETE'])
def eliminar_historial(id):
    registro = Historial.query.get_or_404(id)
    db.session.delete(registro)
    db.session.commit()
    return jsonify({'message': 'Registro eliminado exitosamente'}), 200

@app.route('/api/historial/<int:id>', methods=['PUT'])
def editar_historial(id):
    data = request.get_json()
    registro = Historial.query.get_or_404(id)
    registro.nombre = data.get('nombre', registro.nombre)
    db.session.commit()
    return jsonify({'message': 'Registro editado exitosamente'}), 200

if __name__ == '__main__':
    app.run(debug=True)
