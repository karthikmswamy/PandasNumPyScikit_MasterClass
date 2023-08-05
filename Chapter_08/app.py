from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")


def predict_data(input_data: dict) -> dict:
    # Create a DataFrame with the user input
    test_df = pd.DataFrame(input_data, index=[0])

    [label_encoder_new, std_scaler_new, random_forest_classifier_new, random_forest_regressor_new] = joblib.load('./multi_output_model.pkl')
    test_df_scaled = std_scaler_new.transform(test_df)
    y_pred_style = label_encoder_new.inverse_transform(random_forest_classifier_new.predict(test_df_scaled))
    y_pred_abv = random_forest_regressor_new.predict(test_df_scaled)

    results = {
        "Style": y_pred_style[0],
        "ABV": y_pred_abv[0]
    }

    return results

# Render the HTML form with sliders for user input
@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Handle the form submission and make predictions
@app.post("/predict/")
async def predict_style(min_ibu: float = Form(...),
                        max_ibu: float = Form(...),
                        astringency: float = Form(...),
                        body: float = Form(...),
                        alcohol: float = Form(...),
                        bitter: float = Form(...),
                        sweet: float = Form(...),
                        sour: float = Form(...),
                        salty: float = Form(...),
                        fruits: float = Form(...),
                        hoppy: float = Form(...),
                        spices: float = Form(...),
                        malty: float = Form(...)):
    input_data = {
        'Min IBU': min_ibu,
        'Max IBU': max_ibu,
        'Astringency': astringency,
        'Body': body,
        'Alcohol': alcohol,
        'Bitter': bitter,
        'Sweet': sweet,
        'Sour': sour,
        'Salty': salty,
        'Fruits': fruits,
        'Hoppy': hoppy,
        'Spices': spices,
        'Malty': malty,
    }
    results = predict_data(input_data)

    return results

'''
# Run the FastAPI application with Uvicorn
if __name__ == '__main__':
    data = {
        'Min IBU': 25,
        'Max IBU': 40,
        'Astringency': 32,
        'Body': 27,
        'Alcohol': 5,
        'Bitter': 36,
        'Sweet': 43,
        'Sour': 18,
        'Salty': 7,
        'Fruits': 18,
        'Hoppy': 58,
        'Spices': 5,
        'Malty': 70
    }
    results = predict_data(data)
    print(results)
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
'''