from flask import Flask, request, jsonify
from utils import remove_disfluency
import os

# Initialize Flask server
app = Flask(__name__)

port = int(os.environ.get('PORT', 5000))

# Flash routes
@app.route("/tool/disfluency", methods=["GET", "POST"])

def index():
    if request.method == "POST":
        data = request.get_json()

        if data is None:
            return jsonify({"Error": "Empty request."})

        elif "speech" not in data:
            return jsonify(
                {"Error": "transcription field not found."})

        elif not isinstance(data["speech"], str):
            return jsonify({"Error": "Invalid type of transcription field"})

        try:
            output = remove_disfluency(data['speech'])
            json_output = jsonify(output)
            return jsonify({"speech":output,"words":data["words"]})

        except Exception as exception:
            return jsonify({"Error": str(exception)})

    return "Disfluency Detector API"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
