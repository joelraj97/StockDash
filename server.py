#from flask import Flask, request, jsonify
import predict_model

#app = Flask(__name__)


#@app.route('/predict_share', methods=['GET', 'POST'])
#def predict_share():
    #total_sqft = float(request.form['total_sqft'])
    
#    response = jsonify({
#        'open_high_low_close_ret': predict_model.predict_on_lastupdate().tolist()
#    })
#    response.headers.add('Access-Control-Allow-Origin', '*')

#    return response


#@app.route('/get_date_from_CSV', methods=['GET', 'POST'])

#def get_date_from_CSV():
#    stock_name = (request.form['stock_name'])
    
#    response = jsonify({
#        'last_date_csv_server': predict_model.getdate_csv(stock_name)
#    })
#    response.headers.add('Access-Control-Allow-Origin', '*')

#    return response


#if __name__ == "__main__":
#    print("Starting Python Flask Server For Next day Share Price Prediction...")
    #util.load_saved_artifacts()
#    app.run()



import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output,Input

external_stylesheets = ['app.css']
app=dash.Dash(__name__,external_stylesheets=external_stylesheets)


app.layout = html.Div(style={'background-image':'url("/assets/pngtree-white-smoke-floating-elements-png-image_4158115.jpg")'} ,children=[
#    html.Div(dcc.Input(id='input-on-submit', type='text')),


    dbc.Container([
        html.Button('Predict for Next Day', id='submit-val', n_clicks=0)],
        style={"text-align":"center","font-size":40,"color":"Green"},

          ),
    html.H2("Open",style={"color":"Red","text-align":"center"}),
    html.Br(),
    dbc.Container([
        html.Label(id='Open',className="result",children={})],
         style={"text-align":"center","font-size":40},
       ),
    html.Br(),
    html.H2("High",style={"color":"Red","text-align":"center"}),
    html.Br(),
    dbc.Container([
        html.Label(id='High',className="result",children={})],
         style={"text-align":"center","font-size":20,"color":"Blue"},
    ),
    html.Br(),
    html.H2("Low",style={"color":"Red","text-align":"center"}),
    html.Br(),
    dbc.Container([
        html.Label(id='Low',className="result",children={})],
         style={"text-align":"center","font-size":20,'size': 3,"offset": 4, 'order': 2,"color":"Blue"},
    ),
    html.Br(),
    html.H2("Close",style={"color":"Red","text-align":"center"}),
    html.Br(),
    dbc.Container([
        html.Label(id='Close',className="result",children={})],
         style={"text-align":"center","font-size":20,'size': 3,"offset": 4, 'order': 2,"color":"Blue"},
    ),
    html.Br(),
     html.H2("Date",style={"color":"Red","text-align":"center"}),
     html.Br(),
     dbc.Container([
        html.Label(id='Date',className="result",children={})],
         style={"text-align":"center","font-size":20,'size': 3,"offset": 4, 'order': 2,"color":"Blue"},
    ),
])

@app.callback(
 [ Output(component_id="Open",component_property="children"),
   Output(component_id="High",component_property="children"),
   Output(component_id="Low",component_property="children"),
   Output(component_id="Close",component_property="children"),
   Output(component_id="Date",component_property="children"),
   ],
   Input(component_id="submit-val",component_property="n_clicks"),
)
def update_output(n_clicks):
    pred_return = predict_model.predict_on_lastupdate().tolist()
    opentemp = str(pred_return[0])
    open="Open:"+opentemp
    hightemp=str(pred_return[1])
    high="High:"+ hightemp
    lowtemp=str(pred_return[2])
    low="Low:"+lowtemp
    closetemp=str(pred_return[3])
    close="Close:"+closetemp
    stock_name = "RELIANCE"
    date_return = predict_model.getdate_csv(stock_name)		
    return opentemp,hightemp,lowtemp,closetemp,date_return


if __name__ == '__main__':
	app.run_server(debug=True)

