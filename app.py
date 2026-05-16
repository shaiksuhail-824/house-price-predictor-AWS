from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model package
pkg      = joblib.load('model.pkl')
model    = pkg['model']
scaler   = pkg['scaler']
selector = pkg['selector']

# Exact 50 features selected by SelectKBest during training
FEATURE_COLUMNS = [
    'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
    'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'HalfBath',
    'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF',
    'MSZoning_RL', 'MSZoning_RM', 'LotShape_Reg',
    'Neighborhood_NoRidge', 'Neighborhood_NridgHt',
    'Exterior1st_VinylSd', 'Exterior2nd_VinylSd',
    'MasVnrType_None', 'MasVnrType_Stone',
    'ExterQual_Gd', 'ExterQual_TA',
    'Foundation_CBlock', 'Foundation_PConc',
    'BsmtQual_Gd', 'BsmtQual_TA',
    'BsmtExposure_Gd', 'BsmtFinType1_GLQ',
    'HeatingQC_TA', 'CentralAir_Y', 'Electrical_SBrkr',
    'KitchenQual_Gd', 'KitchenQual_TA',
    'FireplaceQu_Gd', 'FireplaceQu_None',
    'GarageType_Attchd', 'GarageType_Detchd', 'GarageType_None',
    'GarageFinish_None', 'GarageFinish_Unf',
    'GarageQual_None', 'GarageQual_TA',
    'GarageCond_None', 'GarageCond_TA',
    'PavedDrive_Y', 'SaleType_New', 'SaleCondition_Partial'
]

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        # --- Numerical inputs from form ---
        lot_frontage   = float(form.get('LotFrontage', 70))
        lot_area       = float(form.get('LotArea', 8000))
        overall_qual   = float(form.get('OverallQual', 5))
        year_built     = float(form.get('YearBuilt', 2000))
        year_remod     = float(form.get('YearRemodAdd', 2000))
        mas_vnr_area   = float(form.get('MasVnrArea', 0))
        total_bsmt_sf  = float(form.get('TotalBsmtSF', 800))
        gr_liv_area    = float(form.get('GrLivArea', 1500))
        full_bath      = float(form.get('FullBath', 2))
        half_bath      = float(form.get('HalfBath', 0))
        fireplaces     = float(form.get('Fireplaces', 0))
        garage_cars    = float(form.get('GarageCars', 2))
        wood_deck_sf   = float(form.get('WoodDeckSF', 0))
        open_porch_sf  = float(form.get('OpenPorchSF', 0))

        # --- Categorical inputs from form ---
        ms_zoning      = form.get('MSZoning', 'RL')
        lot_shape      = form.get('LotShape', 'Reg')
        neighborhood   = form.get('Neighborhood', 'Other')
        exterior1st    = form.get('Exterior1st', 'Other')
        exterior2nd    = form.get('Exterior2nd', 'Other')
        mas_vnr_type   = form.get('MasVnrType', 'None')
        exter_qual     = form.get('ExterQual', 'TA')
        foundation     = form.get('Foundation', 'Other')
        bsmt_qual      = form.get('BsmtQual', 'TA')
        bsmt_exposure  = form.get('BsmtExposure', 'No')
        bsmt_fin_type1 = form.get('BsmtFinType1', 'Other')
        heating_qc     = form.get('HeatingQC', 'Ex')
        central_air    = form.get('CentralAir', 'Y')
        electrical     = form.get('Electrical', 'SBrkr')
        kitchen_qual   = form.get('KitchenQual', 'TA')
        fireplace_qu   = form.get('FireplaceQu', 'None')
        garage_type    = form.get('GarageType', 'Attchd')
        garage_finish  = form.get('GarageFinish', 'Unf')
        garage_qual    = form.get('GarageQual', 'TA')
        garage_cond    = form.get('GarageCond', 'TA')
        paved_drive    = form.get('PavedDrive', 'Y')
        sale_type      = form.get('SaleType', 'WD')
        sale_condition = form.get('SaleCondition', 'Normal')

        # --- Build feature row with all 50 features set to 0 ---
        row = {col: 0 for col in FEATURE_COLUMNS}

        # Fill numerical (apply log1p to skewed ones like training did)
        row['LotFrontage']  = np.log1p(lot_frontage)
        row['LotArea']      = np.log1p(lot_area)
        row['OverallQual']  = overall_qual
        row['YearBuilt']    = year_built
        row['YearRemodAdd'] = year_remod
        row['MasVnrArea']   = np.log1p(mas_vnr_area)
        row['TotalBsmtSF']  = np.log1p(total_bsmt_sf)
        row['GrLivArea']    = np.log1p(gr_liv_area)
        row['FullBath']     = full_bath
        row['HalfBath']     = half_bath
        row['Fireplaces']   = fireplaces
        row['GarageCars']   = garage_cars
        row['WoodDeckSF']   = np.log1p(wood_deck_sf)
        row['OpenPorchSF']  = np.log1p(open_porch_sf)

        # Fill one-hot encoded categorical features
        if ms_zoning == 'RL': row['MSZoning_RL'] = 1
        if ms_zoning == 'RM': row['MSZoning_RM'] = 1
        if lot_shape == 'Reg': row['LotShape_Reg'] = 1
        if neighborhood == 'NoRidge': row['Neighborhood_NoRidge'] = 1
        if neighborhood == 'NridgHt': row['Neighborhood_NridgHt'] = 1
        if exterior1st == 'VinylSd': row['Exterior1st_VinylSd'] = 1
        if exterior2nd == 'VinylSd': row['Exterior2nd_VinylSd'] = 1
        if mas_vnr_type == 'None':  row['MasVnrType_None'] = 1
        if mas_vnr_type == 'Stone': row['MasVnrType_Stone'] = 1
        if exter_qual == 'Gd': row['ExterQual_Gd'] = 1
        if exter_qual == 'TA': row['ExterQual_TA'] = 1
        if foundation == 'CBlock': row['Foundation_CBlock'] = 1
        if foundation == 'PConc':  row['Foundation_PConc'] = 1
        if bsmt_qual == 'Gd': row['BsmtQual_Gd'] = 1
        if bsmt_qual == 'TA': row['BsmtQual_TA'] = 1
        if bsmt_exposure == 'Gd':  row['BsmtExposure_Gd'] = 1
        if bsmt_fin_type1 == 'GLQ': row['BsmtFinType1_GLQ'] = 1
        if heating_qc == 'TA':  row['HeatingQC_TA'] = 1
        if central_air == 'Y':  row['CentralAir_Y'] = 1
        if electrical == 'SBrkr': row['Electrical_SBrkr'] = 1
        if kitchen_qual == 'Gd': row['KitchenQual_Gd'] = 1
        if kitchen_qual == 'TA': row['KitchenQual_TA'] = 1
        if fireplace_qu == 'Gd':   row['FireplaceQu_Gd'] = 1
        if fireplace_qu == 'None': row['FireplaceQu_None'] = 1
        if garage_type == 'Attchd': row['GarageType_Attchd'] = 1
        if garage_type == 'Detchd': row['GarageType_Detchd'] = 1
        if garage_type == 'None':   row['GarageType_None'] = 1
        if garage_finish == 'None': row['GarageFinish_None'] = 1
        if garage_finish == 'Unf':  row['GarageFinish_Unf'] = 1
        if garage_qual == 'None': row['GarageQual_None'] = 1
        if garage_qual == 'TA':   row['GarageQual_TA'] = 1
        if garage_cond == 'None': row['GarageCond_None'] = 1
        if garage_cond == 'TA':   row['GarageCond_TA'] = 1
        if paved_drive == 'Y':    row['PavedDrive_Y'] = 1
        if sale_type == 'New':         row['SaleType_New'] = 1
        if sale_condition == 'Partial': row['SaleCondition_Partial'] = 1

        # Build DataFrame in exact column order
        input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # Scale using training scaler
        input_scaled = scaler.transform(input_df)

        # Predict (model outputs log price)
        log_pred   = model.predict(input_scaled)[0]
        prediction = int(np.expm1(log_pred))

        return render_template('index.html',
            prediction=f"Predicted Sale Price: ${prediction:,}")

    except Exception as e:
        return render_template('index.html',
            prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)