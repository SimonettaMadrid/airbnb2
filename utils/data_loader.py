import streamlit as st
import pandas as pd

@st.cache_resource
def load_all_data():
    """Carga los CSV de Airbnb desde Google Drive usando sus IDs."""
    
    links = {
        "Barcelona": "1logPE4TVW7ZA9YNrh4797dd_wTaFAisf",
        "Cambridge": "1EVjK_7QJd1tVAlrTP-KHj0niuP9suHCx",
        "Boston": "1qpv-9Vh7DKtb0VZ8ezBuZDUU2BU_L74R",
        "Hawai": "1hlqleiK-0kEcfglzRRS-cwsjfqraxREh",
        "Budapest": "1b1KsI0xS_1T2UrpSZW777aWM64Hu9jm5",
    }

    def gdrive_to_df(file_id):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return pd.read_csv(url)

    dfb = gdrive_to_df(links["Barcelona"])
    dfc = gdrive_to_df(links["Cambridge"])
    dfbo = gdrive_to_df(links["Boston"])
    dfh = gdrive_to_df(links["Hawai"])
    dfbu = gdrive_to_df(links["Budapest"])

    return dfb, dfc, dfbo, dfh, dfbu
