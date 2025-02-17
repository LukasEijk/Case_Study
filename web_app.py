import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score


class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        # Ensure num_heads divides input_dim evenly
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout_linear = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attn(x, x, x)
        attn_output = self.dropout_attn(attn_output)
        x = self.norm1(x + attn_output)

        # Feed forward
        linear_output = self.linear(x)
        linear_output = self.activation(linear_output)
        linear_output = self.dropout_linear(linear_output)
        x = self.norm2(linear_output)

        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_heads, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.block1 = AttentionBlock(input_dim=input_dim, output_dim=hidden_dim1, num_heads=num_heads, dropout=dropout)
        self.block2 = AttentionBlock(input_dim=hidden_dim1, output_dim=hidden_dim2, num_heads=num_heads, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x



popt = np.array([2.63716244e-06, -2.45105259e-04, 1.66339166e-02])
days_cut_off = 30  # or any other appropriate value

def parab(x, a, b, c):
    return a * x**2 + b * x + c

def process_new_entry(new_data, raw_df):
    """
    new_data: dict containing one new customer record with the same columns as raw_df
    raw_df: pandas DataFrame containing the raw data
    
    Returns:
        A DataFrame with one row (the new data entry) with all the processed features.
    """
    # Convert new_data dict into a DataFrame
    new_entry = pd.DataFrame([new_data])
    
    # Append new entry to the raw DataFrame (make a copy to avoid modifying original)
    df = pd.concat([raw_df.copy(), new_entry], ignore_index=True)
    
    # --- Begin processing pipeline ---
    # Filter: Only keep rows with Reisedauer >= 0 (including 0)
    df = df[df.Reisedauer >= 0]
    
    # Map Leistungseintritt to numeric
    df.Leistungseintritt = df.Leistungseintritt.map({'No': 0, 'Yes': 1})
    
    # Replace NaN in Geschlecht with 'X'
    df.Geschlecht = df.Geschlecht.replace({np.nan: 'X'})
    
    # Group by Agenturname to compute booking counts, activations and total commission
    df_plot = df.groupby('Agenturname').agg(
        Buchungen=('Agenturname', 'count'),
        Aktiviert=('Leistungseintritt', 'sum'),
        Ges_Kommission=('Kommission', 'sum')
    ).sort_values(by='Buchungen', ascending=False)
    
    # Calculate Ratio and sort by it (ascending)
    df_plot['Ratio'] = df_plot.Aktiviert / df_plot.Buchungen
    df_plot = df_plot.reset_index().sort_values(by='Ratio', ascending=True).reset_index(drop=True)
    
    # Create Agentur_Rating
    df_plot['Agentur_Rating'] = df_plot.index + 1
    
    # Merge Agentur_Rating and Ges_Kommission back into df
    df = pd.merge(df, df_plot[['Agenturname', 'Agentur_Rating', 'Ges_Kommission']], 
                  on='Agenturname', how='left')
    
    # (Optional) Quick correlation check
    # print('Kurzer Check ob Sinnvoll:')
    # print(df[['Agentur_Rating', 'Leistungseintritt', 'Ges_Kommission']].corr())
    
    # Calculate pred_calc_perc using the parab function for Reisedauer below the days_cut_off threshold
    df['pred_calc_perc'] = np.where(df.Reisedauer < days_cut_off, 
                                    parab(df.Reisedauer, *popt), 0.01)
    
    # Group by Reiseziel to compute travel-related features
    df_reise = df.groupby('Reiseziel').agg(
        Buchungen=('Reiseziel', 'count'),
        Leistungseintritt=('Leistungseintritt', 'sum'),
        Anzahl=('Reiseziel', 'count')
    ).sort_values(by='Buchungen', ascending=False)
    
    df_reise['Reise_Ratio'] = df_reise.Leistungseintritt / df_reise.Buchungen * 100
    df_reise['Reise_oftBesucht'] = np.where(df_reise.Anzahl > 1000., 1, 0)
    
    # Merge travel features back into df
    df = pd.merge(df, df_reise.reset_index()[['Reiseziel', 'Reise_Ratio', 'Reise_oftBesucht']], 
                  on='Reiseziel', how='left')
    
    # Group by Produktname to compute product commission mean
    df_prod = df.groupby('Produktname').agg(Prod_mean=('Kommission', 'mean')).reset_index()
    
    # Merge product features back into df
    df = pd.merge(df, df_prod, on='Produktname', how='left')
    
    # (Optional) Check correlation between Leistungseintritt and Prod_mean
    # print(df[['Leistungseintritt', 'Prod_mean']].corr())
    
    # Create additional binary features from existing columns
    df['is_Airline'] = df.Agenturtyp.map({'Travel Agency': 0, 'Airlines': 1})
    df['is_Online'] = df.Vertriebskanal.map({'Online': 1, 'Offline': 0})
    
    # Create dummy variables for the specified categorical columns
    df = pd.get_dummies(df, columns=['Agenturname', 'Agenturtyp', 'Produktname', 'Reiseziel'])
    df.drop(columns=[ 'Geschlecht', 'Vertriebskanal'], axis=1, inplace=True)
    # --- End processing pipeline ---
    
    # Return the last row (the new entry with all the processed features)
    processed_new_entry = df.tail(1)
    df = df[:-1]
    return processed_new_entry

# ===============================
# Example usage:
# Assuming you have a raw DataFrame named raw_df with the specified columns.
# Here's an example new customer entry:
new_customer = {
    'Agenturname': 'CBH',
    'Agenturtyp': 'Travel Agency',
    'Vertriebskanal': 'Online',
    'Produktname': 'Comprehensive Plan',
    'Reisedauer': 10,
    'Reiseziel': 'MALAYSIA',
    'Nettoumsatz': 1000,
    'Kommission': 50,
    'Geschlecht': 'M',
    'Alter': 35
}
raw_df = pd.read_csv('reiseversicherung.csv')

new_customer = raw_df.loc[1].to_dict()
# Process the new customer:
processed_entry = process_new_entry(new_customer, raw_df)
print(processed_entry)

import streamlit as st
import pandas as pd

def create_input_form():
    # Dummy lists for dropdown menus
    
    agenturname_options = ['CBH', 'CWT', 'JZI', 'KML', 'EPX', 'C2B', 'JWT', 'RAB', 'SSI','ART', 'CSR', 'CCR', 'ADM', 'LWC', 'TTW', 'TST']
    agenturtyp_options = ['Travel Agency', 'Airlines']
    vertriebskanal_options = ['Offline', 'Online']
    produktname_options =['Comprehensive Plan', 'Rental Vehicle Excess Insurance',
       'Value Plan', 'Basic Plan', 'Premier Plan',
       '2 way Comprehensive Plan', 'Bronze Plan', 'Silver Plan',
       'Annual Silver Plan', 'Cancellation Plan',
       '1 way Comprehensive Plan', 'Ticket Protector', '24 Protect',
       'Gold Plan', 'Annual Gold Plan',
       'Single Trip Travel Protect Silver',
       'Individual Comprehensive Plan',
       'Spouse or Parents Comprehensive Plan',
       'Annual Travel Protect Silver',
       'Single Trip Travel Protect Platinum',
       'Annual Travel Protect Gold', 'Single Trip Travel Protect Gold',
       'Annual Travel Protect Platinum', 'Child Comprehensive Plan',
       'Travel Cruise Protect', 'Travel Cruise Protect Family']
    
    reiseziel_options = ['MALAYSIA', 'AUSTRALIA', 'ITALY', 'UNITED STATES', 'THAILAND',
       "'KOREA DEMOCRATIC PEOPLE'S REPUBLIC OF'", 'NORWAY', 'VIET NAM',
       'DENMARK', 'SINGAPORE', 'JAPAN', 'UNITED KINGDOM', 'INDONESIA',
       'INDIA', 'CHINA', 'FRANCE', "'TAIWAN PROVINCE OF CHINA'",
       'PHILIPPINES', 'MYANMAR', 'HONG KONG', "'KOREA REPUBLIC OF'",
       'UNITED ARAB EMIRATES', 'NAMIBIA', 'NEW ZEALAND', 'COSTA RICA',
       'BRUNEI DARUSSALAM', 'POLAND', 'SPAIN', 'CZECH REPUBLIC',
       'GERMANY', 'SRI LANKA', 'CAMBODIA', 'AUSTRIA', 'SOUTH AFRICA',
       "'TANZANIA UNITED REPUBLIC OF'",
       "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 'NEPAL', 'NETHERLANDS',
       'MACAO', 'CROATIA', 'FINLAND', 'CANADA', 'TUNISIA',
       'RUSSIAN FEDERATION', 'GREECE', 'BELGIUM', 'IRELAND',
       'SWITZERLAND', 'CHILE', 'ISRAEL', 'BANGLADESH', 'ICELAND',
       'PORTUGAL', 'ROMANIA', 'KENYA', 'GEORGIA', 'TURKEY', 'SWEDEN',
       'MALDIVES', 'ESTONIA', 'SAUDI ARABIA', 'PAKISTAN', 'QATAR', 'PERU',
       'LUXEMBOURG', 'MONGOLIA', 'ARGENTINA', 'CYPRUS', 'FIJI',
       'BARBADOS', 'TRINIDAD AND TOBAGO', 'ETHIOPIA', 'PAPUA NEW GUINEA',
       'SERBIA', 'JORDAN', 'ECUADOR', 'BENIN', 'OMAN', 'BAHRAIN',
       'UGANDA', 'BRAZIL', 'MEXICO', 'HUNGARY', 'AZERBAIJAN', 'MOROCCO',
       'URUGUAY', 'MAURITIUS', 'JAMAICA', 'KAZAKHSTAN', 'GHANA',
       'UZBEKISTAN', 'SLOVENIA', 'KUWAIT', 'GUAM', 'BULGARIA',
       'LITHUANIA', 'NEW CALEDONIA', 'EGYPT', 'ARMENIA', 'BOLIVIA',
       "'VIRGIN ISLANDS U.S.'", 'PANAMA', 'SIERRA LEONE', 'COLOMBIA',
       'PUERTO RICO', 'UKRAINE', 'GUINEA', 'GUADELOUPE',
       "'MOLDOVA REPUBLIC OF'", 'GUYANA', 'LATVIA', 'ZIMBABWE', 'VANUATU',
       'VENEZUELA', 'BOTSWANA', 'BERMUDA', 'MALI', 'KYRGYZSTAN',
       'CAYMAN ISLANDS', 'MALTA', 'LEBANON', 'REUNION', 'SEYCHELLES',
       'ZAMBIA', 'SAMOA', 'NORTHERN MARIANA ISLANDS', 'NIGERIA',
       'DOMINICAN REPUBLIC', 'TAJIKISTAN', 'ALBANIA',
       "'MACEDONIA THE FORMER YUGOSLAV REPUBLIC OF'F'",
       'LIBYAN ARAB JAMAHIRIYA', 'ANGOLA', 'BELARUS',
       'TURKS AND CAICOS ISLANDS', 'FAROE ISLANDS', 'TURKMENISTAN',
       'GUINEA-BISSAU', 'CAMEROON', 'BHUTAN', 'RWANDA', 'SOLOMON ISLANDS',
       "'IRAN ISLAMIC REPUBLIC OF'", 'GUATEMALA', 'FRENCH POLYNESIA',
       'TIBET', 'SENEGAL', 'REPUBLIC OF MONTENEGRO',
       'BOSNIA AND HERZEGOVINA']
    geschlecht_options = ['M', 'F', np.nan]

    # Create form
    with st.form("new_customer_form"):
        st.header("New Customer Entry Form")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dropdown inputs
            agenturname = st.selectbox(
                "Agenturname",
                options=agenturname_options
            )
            
            agenturtyp = st.selectbox(
                "Agenturtyp",
                options=agenturtyp_options
            )
            
            vertriebskanal = st.selectbox(
                "Vertriebskanal",
                options=vertriebskanal_options
            )
            
            produktname = st.selectbox(
                "Produktname",
                options=produktname_options
            )
            
            geschlecht = st.selectbox(
                "Geschlecht",
                options=geschlecht_options
            )
        
        with col2:
            # Numeric inputs
            reisedauer = st.number_input(
                "Reisedauer",
                min_value=1,
                max_value=15000,
                value=7
            )
            
            reiseziel = st.selectbox(
                "Reiseziel",
                options=reiseziel_options
            )
            
            nettoumsatz = st.number_input(
                "Nettoumsatz",
                value=0.0,
                step=0.01
            )
            
            kommission = st.number_input(
                "Kommission",
                value=0.0,
                step=0.01
            )
            
            alter = st.number_input(
                "Alter",
                min_value=0,
                max_value=120,
                value=30
            )

        # Submit button
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            # Create dictionary with form data
            new_data = {
                'Agenturname': agenturname,
                'Agenturtyp': agenturtyp,
                'Vertriebskanal': vertriebskanal,
                'Produktname': produktname,
                'Leistungseintritt': 'No',
                'Reisedauer': reisedauer,
                'Reiseziel': reiseziel.upper(),  # Convert to uppercase
                'Nettoumsatz': nettoumsatz,
                'Kommission': kommission,
                'Geschlecht': geschlecht,
                'Alter': alter
            }
            
            return new_data, True
    
    return None, False

def main():
    import random
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    raw_df = pd.read_csv('reiseversicherung.csv')
    input_dim = 64  # embedding_dim
    hidden_dim1 = 32
    hidden_dim2 = 16
    num_heads = 4  # Must be a factor of input_dim
    num_classes = 2
    num_epochs = 100
    learning_rate = 1e-4  # Reduced learning rate for stability


    model = TransformerClassifier(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_heads=num_heads,
        num_classes=num_classes,
        dropout=0.1
    )

    model.load_state_dict(torch.load("./MODEL_Training/transformer_classifier_100Epochs.pth",  map_location=torch.device('mps')))

    st.title("Customer Data Entry System")
    
    # Create the form
    new_data, submitted = create_input_form()
    
    # If form was submitted, process the data
    if submitted:
        st.success("Data submitted successfully!")


        
        # Here you would typically call your process_new_entries function
        new_entry = process_new_entry(new_data, raw_df)
        new_entry.drop(columns='Leistungseintritt', axis=1, inplace=True)

        new_entry = new_entry.astype({col: 'int32' for col in new_entry.select_dtypes(include=['bool']).columns})
        new_entry = new_entry.astype({col: 'float32' for col in new_entry.select_dtypes(include=['float64']).columns})
        new_entry = new_entry.astype({col: 'int64' for col in new_entry.select_dtypes(include=['int64']).columns})

        x = torch.tensor(new_entry.values, dtype=torch.float32)  # Shape: (1, 205)

        # Add a feature channel dimension at the end to mimic training shape (batch, num_features, 1)
        x = x.unsqueeze(-1)  # Now x.shape is (1, 205, 1)

        # Define the projection layer (same as used during training)
        input_dim = 64  # embedding dimension
        projection = nn.Linear(1, input_dim)

        with torch.no_grad():
            x = projection(x)  # Now x.shape is (1, 205, 64)

        # Now pass the tensor to your model
        model.eval()
        with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        


        pred_prev = predicted_class.item()
        label_mapping = {0: "Ja", 1: "Nein"}
        predicted_label = label_mapping[predicted_class.item()]
        # st.write("Vorhergesagter Leistungseintritt:", predicted_label)
        # st.write("Vorhergesagte Wahrscheinlichkeit:", np.round(probabilities.detach().numpy().ravel()[pred_prev] * 100,2), '%')

        box_html = f"""
        <div style="
            background-color: #f9f9f9;
            border-left: 5px solid #4CAF50;
            padding: 16px;
            margin: 16px 0;
            border-radius: 5px;
        ">
            <h4 style="margin-bottom: 4px;">Vorhergesagter Leistungseintritt:</h4>
            <p style="font-size: 1.25em; font-weight: bold; margin-top: 0;">{predicted_label}</p>
            <h4 style="margin-bottom: 4px;">Vorhergesagte Wahrscheinlichkeit:</h4>
            <p style="font-size: 1.25em; font-weight: bold; margin-top: 0;">{np.round(probabilities.detach().numpy().ravel()[predicted_class.item()] * 100, 2)}%</p>
        </div>
        """

        st.markdown(box_html, unsafe_allow_html=True)



if __name__ == "__main__":
    main()