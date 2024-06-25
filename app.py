from flask import Flask, request, render_template
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Ruta completa del archivo CSV
csv_path = 'possum.csv'

# Intentar cargar datos desde el CSV
try:
    data = pd.read_csv(csv_path)
    app.logger.debug('Datos cargados correctamente.')

    # Eliminar filas con valores NaN en la columna 'age'
    data = data.dropna(subset=['age'])
    app.logger.debug('Datos limpios, NaN eliminados.')

except FileNotFoundError as e:
    app.logger.error(f'Error al cargar los datos: {str(e)}')
    data = None

# Ruta de la página principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener los datos del formulario y convertir a float
        site = float(request.form['site'])
        hdlngth = float(request.form['hdlngth'])
        skullw = float(request.form['skullw'])
        totlngth = float(request.form['totlngth'])
        footlgth = float(request.form['footlgth'])
        chest = float(request.form['chest'])

        # Realizar la predicción utilizando un modelo (Random Forest Regressor)
        X = data[['site', 'hdlngth', 'skullw', 'totlngth', 'footlgth', 'chest']]
        y = data['age']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Predecir utilizando los datos del formulario
        input_data = np.array([[site, hdlngth, skullw, totlngth, footlgth, chest]])
        predicted_age = int(round(model.predict(input_data)[0]))

        return render_template('index.html', prediction=predicted_age)
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
