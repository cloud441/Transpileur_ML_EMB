import os
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import joblib

model_filename = "lin_reg.sav"

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]


if not (os.path.isfile(model_filename)):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # save the model to disk
    joblib.dump(regr, model_filename)
    model = regr

else:
    # load the model from disk
    model = joblib.load(model_filename)


print(f"coefficients of the model are: {model.coef_}")

c_code = f"#include <stdio.h>\n\n\
float prediction(float feature, int coef, int intercept)\n\
{'{'}\n\
\treturn feature * coef + intercept;\n\
{'}'}\n\n\
int main(void)\n\
{'{'}\n\
\tint coef = {model.coef_[0]};\n\
\tint intercept = {model.intercept_};\n\
\tfloat feature = 0.0779;\n\
\tfloat gt = {model.predict(diabetes_X_test[:1])[0]};\n\
\tprintf(\"prediction is: %f\\nground truth is: %f\\n\", prediction(feature, coef, intercept), gt);\n\
\treturn 0;\n\
{'}'}"


with open("prediction.c", "w") as f:
    f.write(c_code)
