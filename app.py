from flask import Flask
from flask import render_template, request
import re
from calculator.ai_calculator import Calculator
from cabbage.cabbage import Cabbage
from blood.blood import Blood
import _json
import json
import json5
import jsonschema
import jupyterlab_server


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move/<path>')
def mov(path):
    return render_template(f'{path}.html')

@app.route('/calculator')
def ui_calculator():
    stmt = request.args.get('stmt', 'NONE')
    if(stmt == 'NONE'):
        print('넘어온 값이 없음')
    else:
        print(f'넘어온 식: {stmt}')
        patt = '[0-9]+'
        op = re.sub(patt, '', stmt)
        nums = stmt.split(op)
        result = 0
        n1 = int(nums[0])
        n2 = int(nums[1])
        if op == '+': result = n1 + n2
        elif op == '-': result = n1 - n2
        elif op == '*': result = n1 * n2
        elif op == '/': result = n1 / n2
    return jsonify(result = result)

@app.route('/ai_calculator', methods=['POST'])
def ai_calculator():
    num1 = request.form['num1']
    num2 = request.form['num2']
    opcode = request.form['opcode']
    result = Calculator.service(num1, num2, opcode)
    render_params = {}
    render_params['result'] = int(result)
    return render_template('ai_calculator.html',**render_params)

@app.route('/blood', methods=['POST'])
def blood():
    weight = request.form['weight']
    age = request.form['age']
    blood = Blood()
    blood.initialize(weight,age)
    result = blood.service()
    render_params = {}
    render_params['result'] = int(result)
    return render_template('blood.html',**render_params)

@app.route('/cabbage', methods = ['POST'])
def cabbage():
    #avg_temp min_temp max_temp rain_fall
    avg_temp = request.form['avg_temp']#html에서 값을 가져오기
    min_temp = request.form['min_temp']
    max_temp = request.form['max_temp']
    rain_fall = request.form['rain_fall']
    cabbage = Cabbage()
    cabbage.initialize(avg_temp, min_temp, max_temp, rain_fall)
    result = cabbage.service( ) #cabbage 클래스로 값을 보내서 텐서플로우에서 계산된 값을 가져오기
    render_params ={}
    render_params['result'] = int(result)
    return render_template('cabbage.html', **render_params)#html로 결과값을 보내주기


if __name__ == '__main__':
    app.run()