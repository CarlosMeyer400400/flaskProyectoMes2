from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Ruta completa del archivo CSV
csv_path = 'mushroom_cleaned.csv'

# Intentar cargar datos desde el CSV
try:
    data = pd.read_csv(csv_path)
    app.logger.debug('Datos cargados correctamente.')
except FileNotFoundError as e:
    app.logger.error(f'Error al cargar los datos: {str(e)}')
    data = None

# Verifica que los datos fueron cargados correctamente
if data is not None:
    # Asegúrate de eliminar la columna `gill-color` si existe
    if 'gill-color' in data.columns:
        data = data.drop('gill-color', axis=1)
    
    # Separar las características y la etiqueta
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    app.logger.debug('Modelo entrenado correctamente.')
else:
    model = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500

    try:
        # Obtener los datos enviados en el request
        cap_diameter = int(request.form['cap_diameter'])
        cap_shape = int(request.form['cap_shape'])
        gill_attachment = int(request.form['gill_attachment'])
        stem_height = float(request.form['stem_height'])
        stem_width = int(request.form['stem_width'])
        stem_color = int(request.form['stem_color'])
        season = float(request.form['season'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[cap_diameter, cap_shape, gill_attachment, stem_height, stem_width, stem_color, season]],
                               columns=['cap-diameter', 'cap-shape', 'gill-attachment',  'stem-height', 'stem-width', 'stem-color', 'season'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'class': "Comestible" if prediction[0] == 1 else "Venenoso"})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
