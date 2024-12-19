import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_mail():
	textfile = 'time_file.txt'

	# Set up your email parameters
	sender_email = "zacastronomy@gmail.com"
	receiver_email = "zgl12@uclive.ac.nz"  # In this case, it's to yourself
	password = "jzry vrej lflb sbpe"  # For Gmail, you may need to generate an app password

	# Read the contents of the file
	with open(textfile, "r") as file:
		file_content = file.read()

	# Create the email
	subject = "Kakapo Contents of time_file.txt"
	message = MIMEMultipart()
	message["From"] = sender_email
	message["To"] = receiver_email
	message["Subject"] = subject

	# Attach the file content as the email body
	body = MIMEText(file_content, "plain")
	message.attach(body)

	# Connect to Gmail’s SMTP server and send the email
	try:
		with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
			server.login(sender_email, password)
			server.sendmail(sender_email, receiver_email, message.as_string())
			print("Email sent successfully!")
	except Exception as e:
		print(f"Error sending email: {e}")

# send_mail()




