import shap
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

# Feature name mapping for verbose SHAP display
def get_verbose_feature_names(feature_names):
    """Convert model feature names to human-readable names."""
    name_mapping = {
        'CreditScore': 'Credit Score',
        'Geography': 'Geography',
        'Gender': 'Gender',
        'Age': 'Age (years)',
        'Tenure': 'Tenure (years)',
        'Balance': 'Account Balance',
        'NumOfProducts': 'Number of Products',
        'HasCrCard': 'Has Credit Card',
        'IsActiveMember': 'Is Active Member',
        'EstimatedSalary': 'Estimated Salary'
    }
    
    verbose_names = []
    for name in feature_names:
        # Handle one-hot encoded features
        if 'Geography' in name:
            if '_0' in name or name.endswith('0'):
                verbose_names.append('Geography: France')
            elif '_1' in name or name.endswith('1'):
                verbose_names.append('Geography: Spain')
            elif '_2' in name or name.endswith('2'):
                verbose_names.append('Geography: Germany')
            else:
                verbose_names.append(name_mapping.get('Geography', name))
        elif 'Gender' in name:
            if '_0' in name or name.endswith('0'):
                verbose_names.append('Gender: Male')
            elif '_1' in name or name.endswith('1'):
                verbose_names.append('Gender: Female')
            else:
                verbose_names.append(name_mapping.get('Gender', name))
        else:
            verbose_names.append(name_mapping.get(name, name))
    
    return verbose_names

def get_shap_values(explainer: shap.TreeExplainer, processor, df:pd.DataFrame):
    
    # process dataframe
    df_transf = processor.transform(df)

    shap_values = explainer.shap_values(df_transf)[0].tolist()
    feature_names = get_verbose_feature_names(processor.get_feature_names_out().tolist())

    shap_values_dict = {
        "Feature": feature_names,
        "SHAP Impact": shap_values,
    }

    return shap_values_dict

def make_shap_viz(shap_values_dict):

    shap_df = pd.DataFrame(shap_values_dict).sort_values('SHAP Impact', key=abs, ascending=False)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP Impact']]
    ax.barh(shap_df['Feature'], shap_df['SHAP Impact'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=12)
    ax.set_title('Feature Contributions to Churn Prediction\nRed = Increases Churn Risk | Blue = Decreases Churn Risk', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig

def make_beeswarm_viz(explainer: shap.TreeExplainer, processor, df:pd.DataFrame):
    
    df_transf = processor.transform(df)
    shap_values = explainer.shap_values(df_transf)

    shap_df =  pd.DataFrame(
        shap_values,
        columns=get_verbose_feature_names(
            processor.get_feature_names_out()
        )
    )

    plt.figure(figsize=(12,8))
    shap.summary_plot(shap_values, shap_df, plot_type="dot", show=False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    shap_plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return shap_plot_base64


