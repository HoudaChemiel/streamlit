import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter
import pandas as pd
from sklearn.impute import KNNImputer
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import k_fold_cross_validation
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objs as go
import plotly.express as px
import math

st.set_page_config(layout = "wide")

# Charger le jeu de données avec le délimiteur ";"
df = pd.read_csv("mock_data_4.csv", delimiter=";")

dict_trad = {
    'Genre':'Genero', 
    'Réglementation':'Regimenafiliacion',
    'Anémie':'Anemia', 
    'Hypercalcémie' : 'Hipercalcemia', 
    'IRC légère' : 'ERC_Leve', 
    'IRC modérée' : 'ERC_moderada', 
    'IRC sévère' : 'ERC_severa', 
    'IRC dialysée' : 'ERC_dialisis',
    'Lésions osseuses' : 'Lesiones_oseas', 
    'Infections récurrentes' : 'Infecciones_recurrentes', 
    'Fragilité' : 'Fragilidad', 
    'FISH del 17p1' : 'FISHdel17p1', 
    'FISH t(11;14)' : 'FISHt_1114', 
    'FISH t(4;14)' :'FISHt414', 
    'FISH amp(1q21)' : 'FISHamp1q211', 
    'Autres anomalies FISH' : 'FISHother',
    'Sous-classement de la plateforme MM' : 'SubclasificacionplataformaMM', 
    'ISS Plateforme 1' : 'ISSPlataforma1', 
    'Coordonnées osseuses 1' : 'CoadOseo1', 
    'Réponse clinique' : 'RespuestaClinica', 
    'Pays' : 'Country', 
    'Hôpital' : 'Hospital', 
    'Type de myélome' : 'TypeMyeloma',
    'Tranche d\'âge' : 'Age_range', 
    'Evènement' : 'Evento', 
    'Traitement du MM1' : 'TtoMM1', 
    'Traitement du MM2' : 'TtoMM2',
    }

# Menu
selected = option_menu("Analyse de survie", ["Traitement des données manquantes", "Statistiques descriptives", "💰 Analyse coût-efficacité", 
        "Tests de Comparaisons","Probabilités de survie et courbes de survie","Prédiction de survie d'un individu",
        "Modèle de régression de Cox"], 
        icons=['file-spreadsheet-fill', "clipboard-data", "", 'bar-chart-fill',"graph-down","person-bounding-box",
        "graph-up"], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={   
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#8a2be2", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        })
selected

with st.expander('Données brutes'):
    st.write(df)


def proba_survie():
    st.title("Probabilité de Survie")

    # Estimation de la fonction de survie avec Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(df['time'], event_observed=df['Evento'])


    fig_survival = go.Figure()
    fig_survival.add_trace(go.Scatter(x=kmf.timeline, y=kmf.survival_function_.values.flatten(),
                                       mode='lines',
                                       name='Fonction de Survie'))

    fig_survival.update_layout(title='Fonction de Survie (Kaplan-Meier)',
                                xaxis_title='Temps',
                                yaxis_title='Probabilité de survie',
                                hovermode="x unified")

    fig_survival.add_trace(go.Scatter(x=kmf.timeline, y=kmf.confidence_interval_['KM_estimate_upper_0.95'],
                                       line=dict(shape='hv'), mode='lines',
                                       line_color='rgba(0,0,0,0)', showlegend=False))
    fig_survival.add_trace(go.Scatter(x=kmf.timeline, y=kmf.confidence_interval_['KM_estimate_lower_0.95'],
                                       line=dict(shape='hv'), mode='lines',
                                       line_color='rgba(0,0,0,0)', name='Confidence Interval',
                                       fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)'))

    st.plotly_chart(fig_survival, use_container_width=True)

    # Estimation de la fonction de risque avec Nelson-Aalen
    naf = NelsonAalenFitter()
    naf.fit(df['time'], event_observed=df['Evento'])

    fig_hazard = go.Figure()
    fig_hazard.add_trace(go.Scatter(x=naf.cumulative_hazard_.index, y=naf.cumulative_hazard_.values.flatten(),
                                     mode='lines',
                                     name='Fonction de Risque'))

    fig_hazard.update_layout(title='Fonction de Risque (Nelson-Aalen)',
                              xaxis_title='Temps',
                              yaxis_title='Taux de risque',
                              hovermode="x unified")

    fig_hazard.add_trace(go.Scatter(x=naf.cumulative_hazard_.index,
                                     y=naf.confidence_interval_["NA_estimate_upper_0.95"],
                                     line=dict(shape='hv'), mode='lines',
                                     line_color='rgba(0,0,0,0)', showlegend=False))
    fig_hazard.add_trace(go.Scatter(x=naf.cumulative_hazard_.index,
                                     y=naf.confidence_interval_["NA_estimate_lower_0.95"],
                                     line=dict(shape='hv'), mode='lines',
                                     line_color='rgba(0,0,0,0)', name='Confidence Interval',
                                     fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)'))

    st.plotly_chart(fig_hazard, use_container_width=True)
    st.subheader("Comparaison des Fonctions de Survie")
    st.write("Les fonctions de survie de Kaplan-Meier et de Nelson-Aalen peuvent être comparées pour analyser les différences dans la survie.")

    fig_comparison = go.Figure()

    # Fonction de survie de Kaplan-Meier
    fig_comparison.add_trace(go.Scatter(x=kmf.timeline, y=kmf.survival_function_.values.flatten(),
                                         mode='lines',
                                         name='Survie (Kaplan-Meier)',
                                         line=dict(color='blue')))

    # Fonction de survie de Nelson-Aalen
    fig_comparison.add_trace(go.Scatter(x=naf.cumulative_hazard_.index, y=1 - naf.cumulative_hazard_.values.flatten(),
                                         mode='lines',
                                         name='Survie (Nelson-Aalen)',
                                         line=dict(color='red')))

    fig_comparison.update_layout(title='Comparaison des Fonctions de Survie',
                                 xaxis_title='Temps',
                                 yaxis_title='Probabilité de survie',
                                 hovermode="x unified")

    st.plotly_chart(fig_comparison, use_container_width=True)

# Fonction pour le traitement des données manquantes
def traitement_donnees_manquantes():
    st.title("Traitement des données manquantes")

    st.subheader("Informations sur les variables:")
    variables_info = pd.DataFrame({
        "Type de la variable": df.dtypes,
        "Nombre de valeurs manquantes": df.isnull().sum()
    })
    st.write(variables_info)

    selected_columns = st.multiselect("Sélectionner les variables à traiter:", df.columns)

    traitement_method = st.selectbox("Choisir une méthode de traitement:", [
        "Suppression des lignes ou des colonnes",
        "Imputation par la moyenne",
        "Imputation par la médiane",
        "Imputation par la valeur la plus fréquente",
        "Imputation par k-plus proches voisins (KNN)",
        "Saisie manuelle"
    ])

    if traitement_method == "Suppression des lignes ou des colonnes":
        st.subheader("Choisir les actions à effectuer:")
        delete_rows = st.checkbox("Supprimer les lignes avec des données manquantes")
        delete_columns = st.checkbox("Supprimer les colonnes avec des données manquantes")

    if traitement_method == "Saisie manuelle":
        st.subheader("Saisie manuelle d'une valeur pour remplacer les valeurs manquantes:")
        replacement_value = st.text_input("Saisir une valeur de remplacement:")

    if traitement_method in ["Imputation par la moyenne", "Imputation par la médiane", "Imputation par k-plus proches voisins (KNN)"]:
        numerical_columns = df[selected_columns].select_dtypes(include=['number']).columns.tolist()
        if numerical_columns != selected_columns:
            st.warning("L'imputation ne peut être appliquée qu'aux variables numériques.")

    if st.button("Appliquer ce traitement", key="apply_button"):
        if traitement_method == "Suppression des lignes ou des colonnes":
            if delete_rows:
                df.dropna(subset=selected_columns, inplace=True)
            if delete_columns:
                df.dropna(axis=1, inplace=True)
        elif traitement_method == "Imputation par la moyenne" and numerical_columns == selected_columns:
            for column in selected_columns:
                df[column].fillna(df[column].mean(), inplace=True)
        elif traitement_method == "Imputation par la médiane" and numerical_columns == selected_columns:
            for column in selected_columns:
                df[column].fillna(df[column].median(), inplace=True)
        elif traitement_method == "Imputation par la valeur la plus fréquente":
            for column in selected_columns:
                df[column].fillna(df[column].mode()[0], inplace=True)
        elif traitement_method == "Imputation par k-plus proches voisins (KNN)" and numerical_columns == selected_columns:
            imputer = KNNImputer(n_neighbors=5)
            df[selected_columns] = imputer.fit_transform(df[selected_columns])
        elif traitement_method == "Saisie manuelle":
            for column in selected_columns:
                if df[column].dtype == "object" and not replacement_value.isnumeric():
                    st.error(f"La valeur saisie ne correspond pas au type de données de la colonne '{column}'.")
                    return

            for column in selected_columns:
                if replacement_value:
                    df[column].fillna(replacement_value, inplace=True)

        st.subheader("Jeu de données après traitement:")
        st.write(df)

        # Téléchargerment du nouveau jeu de données au format CSV
        st.download_button(
            label="Télécharger le nouveau jeu de données (CSV)",
            data=df.to_csv().encode('utf-8'),
            file_name='imputed_data.csv',
            mime='text/csv',
        )


def test_comparaison():
    selected_cols = ['Genero', 'Regimenafiliacion', 'Anemia', 'Hipercalcemia', 'ERC_Leve', 'ERC_moderada', 'ERC_severa', 'ERC_dialisis',
                     'Lesiones_oseas', 'Infecciones_recurrentes', 'Fragilidad', 'FISHdel17p1', 'FISHt_1114', 'FISHt414', 'FISHamp1q211', 'FISHother',
                     'SubclasificacionplataformaMM', 'ISSPlataforma1', 'CoadOseo1', 'RespuestaClinica', 'Country', 'Hospital', 'TypeMyeloma',
                     'Age_range', 'Evento', 'TtoMM1', 'TtoMM2']

    
    selected_cols_translated = list(dict_trad.keys()) 
    surv_comparison_feature = st.selectbox('Variable à utiliser pour la comparaison de survie', selected_cols_translated)

    colors = [
        (255,   0,   0),
        (  0, 255,   0),
        (  0,   0, 255),
        (255, 128,   0),
        (  0, 255, 128),
        (128,   0, 255),
        (128, 255,   0),
        (  0, 128, 255),
        (255,   0, 128),
        (255, 255,   0),
        (  0, 255, 255),
        (255,   0, 255),
        (128, 128, 128),
    ]

    if st.checkbox('Afficher intervalle de confiance'):
        conf = True
    else:
        conf = False
    
    if st.checkbox('Afficher les courbes sur des grilles distinctes'):
        grid = True
    else:
        grid = False
        
    cat_values = df[dict_trad[surv_comparison_feature]].dropna().unique()
    kmf = KaplanMeierFitter()
    df.loc[df.Evento == 0, 'Dead'] = 0
    df.loc[df.Evento == 1, 'Dead'] = 1
    grid_idcs = None
    if grid:
        if len(cat_values) == 1:
            fig = go.Figure()
            grid_idcs = 1
        elif len(cat_values) == 2:
            fig = make_subplots(rows=1, cols=2, x_title='Durée d\'exposition', y_title='Probabilité de survie')
            grid_idcs = 2
        elif len(cat_values) == 3:
            fig = make_subplots(rows=1, cols=3, x_title='Durée d\'exposition', y_title='Probabilité de survie')
            grid_idcs = 3
        elif len(cat_values) > 3:
            fig = make_subplots(rows=math.ceil(len(cat_values) / 3), cols=3, x_title='Durée d\'exposition', y_title='Probabilité que la maladie ne s\'aggrave pas')
            grid_idcs = 6
    else:
        fig = go.Figure()

    for i, cat_val in enumerate(cat_values):
        c = colors[i % len(colors)]
        translated_feature = dict_trad.get(surv_comparison_feature, surv_comparison_feature)
        flag = (df[dict_trad[surv_comparison_feature]] == cat_val)
        kmf.fit(durations=df[flag]['time'], event_observed=df[flag]['Dead'])
        x = kmf.survival_function_['KM_estimate'].index.values
        y = kmf.survival_function_['KM_estimate']
        y_upper = kmf.confidence_interval_['KM_estimate_upper_0.95']
        y_lower = kmf.confidence_interval_['KM_estimate_lower_0.95']
        row = None
        col = None
        if grid_idcs == 2 or grid_idcs == 3:
            row = 1
            col = i + 1
        if grid_idcs == 6:
            row = (i // 3) + 1
            col = (i % 3) + 1
        if conf:
            fig.add_traces([go.Scatter(x=x, y=y_upper, line=dict(shape='hv'), mode='lines',
                                        line_color='rgba(0,0,0,0)', showlegend=False),
                            go.Scatter(x=x, y=y_lower, line=dict(shape='hv'), mode='lines',
                                        line_color='rgba(0,0,0,0)', showlegend=False,
                                        fill='tonexty', fillcolor=f'rgba({c[0]}, {c[1]}, {c[2]}, 0.2)')], rows=row, cols=col)
        fig.add_traces(go.Scatter(x=x, y=y, line=dict(shape='hv', width=2.5), mode='lines', line_color=f'rgba({c[0]}, {c[1]}, {c[2]}, 1)',
                                name=f'{surv_comparison_feature}={cat_val}'), rows=row, cols=col)

    if not grid or grid_idcs == 1:
        fig.update_layout(
            xaxis_title_text='Durée d\'exposition',
            yaxis_title_text='Probabilité que la maladie ne s\'aggrave pas'
        )
    # else:
    #     fig.update_layout(title_text='Estimacion de curvas de sobrevida')
    st.subheader('Estimation de la survie : Kaplan Meier')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")

    # Nelson Aalen
    naf = NelsonAalenFitter()
    grid_idcs = None
    if grid:
        if len(cat_values) == 1:
            fig = go.Figure()
            grid_idcs = 1
        elif len(cat_values) == 2:
            fig = make_subplots(rows=1, cols=2, x_title='time', y_title='Hazard')
            grid_idcs = 2
        elif len(cat_values) == 3:
            fig = make_subplots(rows=1, cols=3, x_title='time', y_title='Hazard')
            grid_idcs = 3
        elif len(cat_values) > 3:
            fig = make_subplots(rows=math.ceil(len(cat_values) / 3), cols=3, x_title='time', y_title='Hazard')
            grid_idcs = 6
    else:
        fig = go.Figure()
    for i, cat_val in enumerate(cat_values):
        c = colors[i % len(colors)]
        flag = (df[dict_trad[surv_comparison_feature]] == cat_val)
        naf.fit(durations=df[flag]['time'], event_observed=df[flag]['Dead'])
        x = naf.cumulative_hazard_['NA_estimate'].index.values
        y = naf.cumulative_hazard_['NA_estimate']
        y_upper = naf.confidence_interval_['NA_estimate_upper_0.95']
        y_lower = naf.confidence_interval_['NA_estimate_lower_0.95']
        row = None
        col = None
        if grid_idcs == 2 or grid_idcs == 3:
            row = 1
            col = i + 1
        if grid_idcs == 6:
            row = (i // 3) + 1
            col = (i % 3) + 1
        if conf:
            fig.add_traces([go.Scatter(x=x, y=y_upper, line=dict(shape='hv'), mode='lines',
                                        line_color='rgba(0,0,0,0)', showlegend=False),
                            go.Scatter(x=x, y=y_lower, line=dict(shape='hv'), mode='lines',
                                        line_color='rgba(0,0,0,0)', showlegend=False,
                                        fill='tonexty', fillcolor=f'rgba({c[0]}, {c[1]}, {c[2]}, 0.2)')], rows=row, cols=col)
        fig.add_traces(go.Scatter(x=x, y=y, line=dict(shape='hv', width=2.5), mode='lines', line_color=f'rgba({c[0]}, {c[1]}, {c[2]}, 1)',
                                name=f'{surv_comparison_feature}={cat_val}'), rows=row, cols=col)
    if not grid or grid_idcs == 1:
        fig.update_layout(
            xaxis_title_text='Temps',
            yaxis_title_text='Risque'
        )
    # else:
    #     fig.update_layout(title_text='Nelson Aalen')
    st.subheader('Analyse de Nelson Aalen (risque cumulé)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")

    # Weibull
    wf = WeibullFitter()
    grid_idcs = None
    if grid:
        if len(cat_values) == 1:
            fig = go.Figure()
            grid_idcs = 1
        elif len(cat_values) == 2:
            fig = make_subplots(rows=1, cols=2, x_title='Temps', y_title='Risque')
            grid_idcs = 2
        elif len(cat_values) == 3:
            fig = make_subplots(rows=1, cols=3, x_title='Temps', y_title='Risque')
            grid_idcs = 3
        elif len(cat_values) > 3:
            fig = make_subplots(rows=math.ceil(len(cat_values) / 3), cols=3, x_title='Temps', y_title='Risque')
            grid_idcs = 6
    else:
        fig = go.Figure()

    for i, cat_val in enumerate(cat_values):
        c = colors[i % len(colors)]
        flag = (df[dict_trad[surv_comparison_feature]] == cat_val)
        wf.fit(durations=df[flag]['time'].astype(float), event_observed=df[flag]['Dead'].astype(float))
        x = wf.cumulative_hazard_['Weibull_estimate'].index.values
        y = wf.cumulative_hazard_['Weibull_estimate']
        y_upper = wf.confidence_interval_['Weibull_estimate_upper_0.95']
        y_lower = wf.confidence_interval_['Weibull_estimate_lower_0.95']
        row = None
        col = None
        if grid_idcs == 2 or grid_idcs == 3:
            row = 1
            col = i + 1
        if grid_idcs == 6:
            row = (i // 3) + 1
            col = (i % 3) + 1
        if conf:
            fig.add_traces([go.Scatter(x=x, y=y_upper, line=dict(shape='hv'), mode='lines',
                                        line_color='rgba(0,0,0,0)', showlegend=False),
                            go.Scatter(x=x, y=y_lower, line=dict(shape='hv'), mode='lines',
                                        line_color='rgba(0,0,0,0)', showlegend=False,
                                        fill='tonexty', fillcolor=f'rgba({c[0]}, {c[1]}, {c[2]}, 0.2)')], rows=row, cols=col)
        fig.add_traces(go.Scatter(x=x, y=y, line=dict(shape='hv', width=2.5), mode='lines', line_color=f'rgba({c[0]}, {c[1]}, {c[2]}, 1)',
                                name=f'{surv_comparison_feature}={cat_val}'), rows=row, cols=col)
    if not grid or grid_idcs == 1:
        fig.update_layout(
            xaxis_title_text='Temps',
            yaxis_title_text='Risque'
        )
    # else:
        # fig.update_layout(title_text='Weibull')
    st.subheader('Analyse de Weibull (risque cumulé)')
    st.plotly_chart(fig, use_container_width=True)


# Régression de Cox
def cox_regression():
    st.title("Analyse de Régression de Cox")

    st.subheader("Sélection des covariables :")
    cox_variables = st.multiselect('Sélectionnez les covariables :', df.columns)

    if not cox_variables:
        st.warning("Veuillez sélectionner des covariables pour procéder à l'analyse de régression de Cox.")
        return

    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col='time', event_col='Evento', formula=' + '.join(cox_variables))
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de l'entraînement du modèle : {e}")
        return

    st.subheader('Paramètres du modèle de Cox :')
    st.write(cph.params_)

    # Calcul des p-valeurs
    st.subheader('Valeurs de p :')
    p_values = cph.summary.p
    tmp_df = pd.DataFrame(p_values)
    tmp_df['Signification'] = np.where(tmp_df['p'] < 0.05, 'Impact significatif sur la survie', 'Aucun impact significatif sur la survie')
    st.write(tmp_df)

    st.subheader('Interprétation des coefficients :')
    st.write("Les coefficients positifs indiquent un risque accru, tandis que les coefficients négatifs indiquent un risque réduit.")

    
    st.subheader('Graphique en forêt :')
    try:
        fig, ax = plt.subplots()
        cph.plot(ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du tracé des courbes de survie : {e}")

    st.subheader('Rapports de risques :')
    ratios = cph.hazard_ratios_
    st.write(ratios)
    max_impact_var = list(ratios.nlargest(1).index)[0]
    st.write(f"La variable '{max_impact_var}' a le plus grand impact sur le décès.")

    st.subheader('Résultats détaillés :')
    st.write(cph.summary)

    try:
        residuals = cph.compute_residuals(df, kind='scaled_schoenfeld')       
        #residuals_df = pd.concat([df['time'], residuals], axis=1)
        #residuals_df.columns = ['time', 'residuals']
        
        #fig, ax = plt.subplots()
        #for i, col in enumerate(residuals_df.columns[1:]):
        #    ax.scatter(residuals_df['time'], residuals_df[col], label=col)
        #ax.axhline(y=0, color='r', linestyle='-')
        #ax.set_xlabel('Time')
        #ax.set_ylabel('Residuals')
        #ax.set_title('Residuals vs Time')
        #ax.legend()
        #st.pyplot(fig)
    except Exception as e:
        st.error(f"Error occurred while plotting residuals vs time: {e}")

def cost__analysis():
    st.title("Analyse Coût-Efficacité")
    traitement = st.selectbox('Sélectionnez un traitement:', df['TtoMM1'].unique())

    donnees_filtrees = df[df['TtoMM1'] == traitement]
    pays = st.selectbox('Sélectionnez un pays:', donnees_filtrees['Country'].unique())

    donnees_filtrees = donnees_filtrees[donnees_filtrees['Country'] == pays]

    hopital = st.selectbox('Sélectionnez un hôpital:', donnees_filtrees['Hospital'].unique())

    donnees_filtrees = donnees_filtrees[donnees_filtrees['Hospital'] == hopital]

    # Statistiques descriptives
    cout_moyen = donnees_filtrees['Cost'].mean()
    cout_median = donnees_filtrees['Cost'].median()
    cout_std = donnees_filtrees['Cost'].std()

    st.subheader('Statistiques descriptives :')
    st.write(f'Coût moyen du traitement : {cout_moyen:.2f} $')
    st.write(f'Médiane du coût du traitement : {cout_median:.2f} $')
    st.write(f'Écart-type du coût du traitement : {cout_std:.2f} $')

    st.subheader('Comparaison avec d\'autres traitements :')
    autres_traitements = st.multiselect('Sélectionnez d\'autres traitements pour la comparaison :', df['TtoMM1'].unique())
    if autres_traitements:
        comparaison_df = df[df['TtoMM1'].isin([traitement] + autres_traitements)]
        fig, ax = plt.subplots()
        sns.boxplot(data=comparaison_df, x='TtoMM1', y='Cost', ax=ax)
        ax.set_xlabel('Traitement')
        ax.set_ylabel('Coût du traitement ($)')
        st.pyplot(fig)



def preprocess_data(df):
    # Convertion des variables catégorielles en variables indicatrices (One-Hot Encoding)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def survival_prediction(df):
    st.title("Prédiction de survie")

    model_choice = st.radio("Choisissez un modèle :", ("Kaplan-Meier", "Weibull"))
    st.subheader("Sélectionnez les covariables pour la prédiction:")
    covariables = st.multiselect('Sélectionnez les covariables:', df.columns)

    jours = st.number_input("Nombre de jours:", min_value=1)

    show_ci = st.checkbox("Afficher l'intervalle de confiance", value=True)

    if st.button("Effectuer la prédiction"):
        if len(covariables) < 1:
            st.error("Veuillez sélectionner au moins une covariable.")
            return

        # Filtrer le dataframe
        data = df[covariables + ["time", "Evento"]].dropna()

        if len(data) < 1:
            st.error("Le dataframe filtré est vide. Veuillez sélectionner d'autres covariables.")
            return

        try:
            data = preprocess_data(data)

            if model_choice == "Kaplan-Meier":
                # Modèle Kaplan-Meier
                kmf = KaplanMeierFitter()
                kmf.fit(data["time"], event_observed=data["Evento"])

                # Estimation de la survie pour le nombre de jours saisi
                survival_proba = kmf.predict(jours)
                st.write("Estimation de la survie pour", jours, "jours avec le modèle Kaplan-Meier:")
                st.write(survival_proba)

                # Visualisation
                st.write("### Courbe de survie Kaplan-Meier")
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=kmf.timeline, y=kmf.survival_function_.values.flatten(),
                                         mode='lines',
                                         name='Courbe de survie'))

                if show_ci:
                    lower_bound = kmf.confidence_interval_['KM_estimate_lower_0.95']
                    upper_bound = kmf.confidence_interval_['KM_estimate_upper_0.95']
                    fig.add_trace(go.Scatter(x=np.concatenate((kmf.timeline, kmf.timeline[::-1])),
                                             y=np.concatenate((lower_bound, upper_bound[::-1])),
                                             fill='toself',
                                             fillcolor='rgba(0,100,80,0.2)',
                                             line=dict(color='rgba(255,255,255,0)'),
                                             showlegend=False,
                                             name='Intervalle de confiance'))

                # Mise en forme de la figure
                fig.update_layout(title="Courbe de survie Kaplan-Meier",
                                  xaxis_title="Temps (jours)",
                                  yaxis_title="Probabilité de survie",
                                  hovermode="x unified")

                # Affichage de la figure
                st.plotly_chart(fig)

                # Tableau de survie
                st.write("### Tableau de survie Kaplan-Meier")
                survival_table = kmf.event_table
                st.write(survival_table)

            elif model_choice == "Weibull":
                # Instance du modèle WeibullFitter
                wf = WeibullFitter()

                try:
                    wf.fit(durations=df['time'].astype(float), event_observed=df['Evento'].astype(float))
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de l'entraînement du modèle Weibull : {e}")
                    return

                st.subheader('Paramètres du modèle de Weibull :')
                st.write(wf.summary)

                # Estimations de la fonction de survie
                st.subheader('Estimations de la fonction de survie :')
                st.write(wf.survival_function_)

                # Estimations de la fonction de risque cumulatif
                st.subheader('Estimations de la fonction de risque cumulatif :')
                st.write(wf.cumulative_hazard_)

                st.subheader('Graphique de la fonction de survie :')
                fig_survival = go.Figure()

                # Courbe de survie
                fig_survival.add_trace(go.Scatter(x=wf.survival_function_.index, y=wf.survival_function_.values.flatten(),
                                                mode='lines', name='Courbe de survie'))

                # Intervalle de confiance
                fig_survival.add_trace(go.Scatter(x=wf.confidence_interval_.index, y=wf.confidence_interval_.values[:, 0],
                                                fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig_survival.add_trace(go.Scatter(x=wf.confidence_interval_.index, y=wf.confidence_interval_.values[:, 1],
                                                fill='tonexty', mode='lines', line_color='rgba(0, 0, 255, 0.2)',
                                                name='Confidence Interval'))

                fig_survival.update_layout(
                    title='Graphique de la fonction de survie',
                    xaxis_title='Time',
                    yaxis_title='Survival Probability'
                )

                st.plotly_chart(fig_survival, use_container_width=True)
                st.subheader('Graphique de la fonction de risque cumulatif :')
                fig_cumulative_hazard = go.Figure()

                fig_cumulative_hazard.add_trace(go.Scatter(x=wf.cumulative_hazard_.index, y=wf.cumulative_hazard_.values.flatten(),
                                                        mode='lines', name='Cumulative Hazard'))

                fig_cumulative_hazard.add_trace(go.Scatter(x=wf.confidence_interval_.index, y=wf.confidence_interval_["Weibull_estimate_lower_0.95"],
                                                        fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig_cumulative_hazard.add_trace(go.Scatter(x=wf.confidence_interval_.index, y=wf.confidence_interval_["Weibull_estimate_upper_0.95"],
                                                        fill='tonexty', mode='lines', line_color='rgba(0, 0, 255, 0.2)',
                                                        name='Confidence Interval'))

                fig_cumulative_hazard.update_layout(
                    title='Graphique de la fonction de risque cumulatif',
                    xaxis_title='Time',
                    yaxis_title='Cumulative Hazard'
                )

                st.plotly_chart(fig_cumulative_hazard, use_container_width=True)

                st.subheader('Graphique de la fonction de risque instantané :')
                fig_hazard = go.Figure()

                fig_hazard.add_trace(go.Scatter(x=wf.hazard_.index, y=wf.hazard_.values.flatten(),
                                                mode='lines', name='Instantaneous Hazard'))

                fig_hazard.add_trace(go.Scatter(x=wf.confidence_interval_.index, y=wf.confidence_interval_["Weibull_estimate_lower_0.95"],
                                                fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig_hazard.add_trace(go.Scatter(x=wf.confidence_interval_.index, y=wf.confidence_interval_["Weibull_estimate_upper_0.95"],
                                                fill='tonexty', mode='lines', line_color='rgba(0, 0, 255, 0.2)',
                                                name='Confidence Interval'))

                fig_hazard.update_layout(
                    title='Graphique de la fonction de risque instantané',
                    xaxis_title='Time',
                    yaxis_title='Instantaneous Hazard'
                )

                st.plotly_chart(fig_hazard, use_container_width=True)

                st.subheader('Graphique des résidus :')
                fig_residuals = go.Figure()

                st.subheader('Tableau des événements observés :')
                st.write(wf.event_table)

        except Exception as e:
            st.error("Une erreur s'est produite lors de la prédiction. Veuillez vérifier vos données et réessayer.")
            st.error(str(e))


import streamlit as st
import plotly.express as px


def descriptive_statistics():

    st.title("Analyse Statistique Descriptive")

    list_option_descriptives_label = list(dict_trad.keys())
    choice = st.selectbox("Sélectionnez une variable :", options=list_option_descriptives_label)

    st.subheader("Statistiques Descriptives :")
    stats = df[dict_trad.get(choice)].describe()
    st.write(stats)

    st.subheader(f"Visualisation Graphique de '{choice}' :")
    if 'mean' in stats.keys():
        # Histogramme 
        range_ = stats['max'] - stats['min']
        fig_hist = px.histogram(x=df[dict_trad.get(choice)], range_x=(stats['min'], stats['max'] + 0.05 * range_))
        fig_hist.update_traces(xbins=dict(start=stats['min'], end=stats['max'] + 0.05 * range_, size=0.05 * range_))
        fig_hist.update_layout(
            xaxis_title_text=choice,
            yaxis_title_text='Count',
            title=f'Histogramme de {choice}'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        # Diagramme à barres
        x = df[dict_trad.get(choice)].unique()
        y_temp = df[dict_trad.get(choice)].value_counts(sort=False)
        y = [y_temp[val] if val in y_temp.keys() else 0 for val in x]
        fig_bar = px.bar(x=x, y=y)
        fig_bar.update_layout(
            xaxis_title_text=choice,
            yaxis_title_text='Count',
            title=f'Diagramme à Barres de {choice}'
        )
        st.plotly_chart(fig_bar, use_container_width=True)


if selected == "Traitement des données manquantes":
    traitement_donnees_manquantes()
if selected == "Modèle de régression de Cox":
    cox_regression()
if selected == "Prédiction de survie d'un individu":
    survival_prediction(df)
if selected == "Statistiques descriptives":
    descriptive_statistics()
if selected == "Tests de Comparaisons":
    test_comparaison()
if selected == "Probabilités de survie et courbes de survie":
    proba_survie()
if selected == "💰 Analyse coût-efficacité":
    cost__analysis()



st.sidebar.title("Filtres de Sélection des Variables")

genero = st.sidebar.multiselect("Genre", df["Genero"].unique().tolist(), df["Genero"].unique().tolist(), key="select_genero")
regimen_afiliacion = st.sidebar.multiselect("Régime d'affiliation", df["Regimenafiliacion"].unique().tolist(), df["Regimenafiliacion"].unique().tolist(), key="select_regimen_afiliacion")
pays = st.sidebar.multiselect("Pays", df["Country"].unique().tolist(), df["Country"].unique().tolist(), key="select_country")
hospital = st.sidebar.multiselect("Hôpital", df["Hospital"].unique().tolist(), df["Hospital"].unique().tolist(), key="select_hospital")
TtoMM1 = st.sidebar.multiselect("TtoMM1", df["TtoMM1"].unique().tolist(), df["TtoMM1"].unique().tolist(), key="select_TtoMM1")
TtoMM2 = st.sidebar.multiselect("TtoMM2", df["TtoMM2"].unique().tolist(), df["TtoMM2"].unique().tolist(), key="select_TtoMM2")
TypeMyeloma = st.sidebar.multiselect("Type de myélome", df["TypeMyeloma"].unique().tolist(), df["TypeMyeloma"].unique().tolist(), key="select_TypeMyeloma")
selected_response_clinique = st.sidebar.multiselect("Réponse clinique", df["RespuestaClinica"].unique().tolist(), df["RespuestaClinica"].unique().tolist(), key="select_RespuestaClinica")

imc_range = st.sidebar.slider("IMC", float(df["IMC"].min()), float(df["IMC"].max()), (float(df["IMC"].min()), float(df["IMC"].max())), key="select_IMC")
adherencia_range = st.sidebar.slider("Adhérence au traitement", float(df["AdherenciaTtoMM1Mto1"].min()), float(df["AdherenciaTtoMM1Mto1"].max()), (float(df["AdherenciaTtoMM1Mto1"].min()), float(df["AdherenciaTtoMM1Mto1"].max())), key="select_AdherenciaTtoMM1Mto1")
oportunidad_range = st.sidebar.slider("Opportunité de traitement", float(df["Oportunidadtratamiento"].min()), float(df["Oportunidadtratamiento"].max()), (float(df["Oportunidadtratamiento"].min()), float(df["Oportunidadtratamiento"].max())), key="select_Oportunidadtratamiento")
dias_suspendidos_range = st.sidebar.slider("Jours suspendus", float(df["Dias_suspendidos"].min()), float(df["Dias_suspendidos"].max()), (float(df["Dias_suspendidos"].min()), float(df["Dias_suspendidos"].max())), key="select_Dias_suspendidos")
dias_en_terapia_range = st.sidebar.slider("Jours en thérapie", float(df["Dias_en_terapia"].min()), float(df["Dias_en_terapia"].max()), (float(df["Dias_en_terapia"].min()), float(df["Dias_en_terapia"].max())), key="select_Dias_en_terapia")
time_before_hospitalisation_range = st.sidebar.slider("Temps avant hospitalisation", float(df["time_before_hospitalisation"].min()), float(df["time_before_hospitalisation"].max()), (float(df["time_before_hospitalisation"].min()), float(df["time_before_hospitalisation"].max())), key="select_time_before_hospitalisation")
age_range = st.sidebar.slider("Âge", float(df["Age"].min()), float(df["Age"].max()), (float(df["Age"].min()), float(df["Age"].max())), key="select_age")
cost_range = st.sidebar.slider("Coût du traitement", float(df["Cost"].min()), float(df["Cost"].max()), (float(df["Cost"].min()), float(df["Cost"].max())), key="select_cost")

#fecha_respuesta_clinica = st.sidebar.date_input("Réponse clinique")
#date_diagnosis_range = st.sidebar.date_input("Date de diagnostic", [pd.to_datetime("1900-01-01"), pd.to_datetime("2100-01-01")])

# Appliquer les filtres
filtered_df = df.copy()

# Appliquer les filtres de sélection multiple
filtered_df = filtered_df[filtered_df["Genero"].isin(genero)]
filtered_df = filtered_df[filtered_df["Regimenafiliacion"].isin(regimen_afiliacion)]
filtered_df = filtered_df[filtered_df["Country"].isin(pays)]
filtered_df = filtered_df[filtered_df["Hospital"].isin(hospital)]
filtered_df = filtered_df[filtered_df["TtoMM1"].isin(TtoMM1)]
filtered_df = filtered_df[filtered_df["TtoMM2"].isin(TtoMM2)]
filtered_df = filtered_df[filtered_df["TypeMyeloma"].isin(TypeMyeloma)]
filtered_df = filtered_df[filtered_df["RespuestaClinica"].isin(selected_response_clinique)]

# Appliquer les filtres de plage
filtered_df = filtered_df[filtered_df["IMC"].between(imc_range[0], imc_range[1])]
filtered_df = filtered_df[filtered_df["AdherenciaTtoMM1Mto1"].between(adherencia_range[0], adherencia_range[1])]
filtered_df = filtered_df[filtered_df["Oportunidadtratamiento"].between(oportunidad_range[0], oportunidad_range[1])]
filtered_df = filtered_df[filtered_df["Dias_suspendidos"].between(dias_suspendidos_range[0], dias_suspendidos_range[1])]
filtered_df = filtered_df[filtered_df["Dias_en_terapia"].between(dias_en_terapia_range[0], dias_en_terapia_range[1])]
filtered_df = filtered_df[filtered_df["time_before_hospitalisation"].between(time_before_hospitalisation_range[0], time_before_hospitalisation_range[1])]
filtered_df = filtered_df[filtered_df["Age"].between(age_range[0], age_range[1])]
filtered_df = filtered_df[filtered_df["Cost"].between(cost_range[0], cost_range[1])]

# Afficher les données filtrées
st.dataframe(filtered_df)