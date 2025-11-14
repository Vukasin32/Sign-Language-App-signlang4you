import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import os

def send_video_email(recipient, video_path):
    try:
        sender = "signlang4you@gmail.com"
        app_password = "gctu qdas adoo tudq"  

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = "Tvoj poklon snimak iz web aplikacije signlang4you"

        body = "Zdravo! \nU prilogu se nalazi poklon video za tebe! \nOvaj video je nastao tokom sesije na web aplikaciji signlang4you."
        msg.attach(MIMEText(body, "plain"))

        with open(video_path, "rb") as file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(video_path)}")
            msg.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)

        print("✅ Email uspešno poslat!")
        return True
    except Exception as e:
        print("❌ Greška pri slanju emaila:", e)
        return False
