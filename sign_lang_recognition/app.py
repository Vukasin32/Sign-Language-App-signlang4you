from flask import Flask, render_template, request, redirect, url_for
import subprocess
from utils.send_email import send_video_email

app = Flask(__name__)

VIDEO_PATH = "static/videos/session_recording.mp4"

# Podrazumevana ruta
@app.route('/')
def index():
    return render_template('index.html')

# Ruta na kojoj se pokreće make_sign_messages.py - skripta za pravljenje video sesije pokzaivanja znakovnog jezika
@app.route('/start')
def start_session():
    subprocess.run(["python", "make_sign_messages.py"])
    return redirect(url_for('finish'))

# Ruta na kojoj se pokreće learn_sign_lang.py - skripta za učenje znakovnog jezika
@app.route('/learn_sign')
def learn_sign():
    subprocess.Popen(["python", "learn_sign_lang.py"])
    return '', 204 

# Ruta sa koje korinsik može da pošalje snimak svoje video sesije na željenu mejl adresu
@app.route('/finish', methods=['GET', 'POST'])
def finish():
    if request.method == 'POST':
        email = request.form.get('email')
        if email:
            success = send_video_email(recipient=email, video_path=VIDEO_PATH)
            if success:
                return render_template('success.html', message="🎉 Video je uspešno poslat!")
            else:
                return render_template('success.html', message="⚠️ Došlo je do greške pri slanju videa.")
        else:
            return render_template('success.html', message="✅ Sesija završena bez slanja videa.")
    return render_template('finish.html')

if __name__ == '__main__':
    app.run(debug=True)
