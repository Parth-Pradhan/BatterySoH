from flask import Flask, jsonify, render_template, request, redirect, url_for
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from analysis import Battery_name, index, n_pred, pred_ahead, dataset, capacity_data, capacity_actual, capacity_pred, score, scores
app2 = Flask(__name__)

@app2.route('/')
def predic():
    
    # Graph one
    fig1 = px.scatter(dataset, x = 'cycle', y = 'capacity', title = "Relationship between discharge cycle and battery capacity")
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph two
    fig2 = px.scatter(capacity_data, x = 'current_measured', y = 'capacity', title = "Relationship between battery output current and battery capacity")
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph three
    fig3 = px.scatter(capacity_data, x = 'voltage_measured', y = 'capacity', title = "Relationship between battery terminal voltage and battery capacity")
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)


    # Graph four
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
    x=[i for i in range(index + pred_ahead - n_pred)],
    y=list(capacity_actual)/capacity_data['capacity'][0],
    name="Actual"       
    ))
    fig4.add_trace(go.Scatter(
    x=[i for i in range(index + pred_ahead - n_pred)],
    y= list(capacity_pred)/capacity_data['capacity'][0],
    name="Predicted"
    ))

    fig4.update_layout(
    title="Actual SoH vs predicted SoH. Prediction horizon: (" + str(index - n_pred) + "," + str(index - n_pred + pred_ahead) + ") and RMSE is " + str(score),
    xaxis_title="Cycle",
    yaxis_title="State of Health (SoH)",
    legend_title="Legends",
    )

    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('results.html', 
    graph1JSON=graph1JSON,  graph2JSON=graph2JSON, graph3JSON=graph3JSON, graph4JSON=graph4JSON)

if __name__ == "__main__":
    app2.run(host ='127.0.0.1', port = '5001', debug = True)




