# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import main
# import numpy as np

# # Assuming forwardPropagate function is defined elsewhere
# # from some_module import forwardPropagate

# network = None

# app = Flask(__name__, static_folder='showcase/showcase_react/build', static_url_path='')
# CORS(app)

# # def forwardPropagate(data):
# #     # Dummy implementation of forwardPropagate
# #     # Replace with your actual implementation
# #     return sum(sum(row) for row in data)

# @app.route('/api', methods=['POST'])
# def handle_api():
#     data = request.json.get('grid', [])
#     if not data:
#         return jsonify({'error': 'No grid data provided'}), 400

#     # Call the forwardPropagate function with the grid data
#     np_data = np.atleast_2d(np.array(data).flatten()).transpose()



#     result = int(np.argmax(network.forward_pass(np_data)))
#     print(result)
#     # print(result)

#     # Return the result as a JSON response
#     return jsonify({'result': result})

# @app.route('/')
# def serve():
#     return send_from_directory(app.static_folder, 'index.html')

# if __name__ == '__main__':
#     network = main.get_best_network()
#     print("Running server on port 5000")
#     app.run(port=5000)



from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import main
import numpy as np

network = None
app = Flask(__name__, static_folder='showcase/showcase_react/build', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api', methods=['POST'])
def handle_api():
    data = request.json.get('grid', [])
    if not data:
        return jsonify({'error': 'No grid data provided'}), 400

    result = network.forward_pass(np.atleast_2d(np.array(data).flatten()).transpose()).tolist()
    return jsonify({'result': result})

@socketio.on('submit_grid')
def handle_submit_grid(data):
    grid = data.get('grid', [])
    result = network.forward_pass(np.atleast_2d(np.array(grid).flatten()).transpose()).tolist()
    emit('result', {'result': result})

if __name__ == '__main__':
    network = main.get_best_network()
    socketio.run(app, port=5000)

