import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Load the saved CNN model
model = load_model('cnn_classification_model1.h5')

# Class labels (update these according to your dataset)
class_labels = {0: 'facade', 1: 'foundation', 2: 'interior', 3: 'superstructure'}

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to the target size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Generate construction report based on the classification
def generate_construction_report(project_name, stage, accuracy_percentage):
    # Define key insights, recommendations, and next steps for each stage
    if stage == "foundation":
        key_insights = f"""
        • Foundation: The image has been identified with {accuracy_percentage:.2f}% accuracy as showing the foundation stage. 
          Ensure the foundation is stable and properly aligned before proceeding to the next construction phase.
        """
        recommendations = """
        • Continue monitoring the foundation quality and ensure it complies with the design specifications.
        """
        next_steps = "Proceed with superstructure construction after confirming the foundation's stability."

    elif stage == "superstructure":
        key_insights = f"""
        • Superstructure: The image has been identified with {accuracy_percentage:.2f}% accuracy as showing the superstructure stage.
          Ensure the vertical alignment and structural integrity of the superstructure before proceeding further.
        """
        recommendations = """
        • Inspect all critical load-bearing points in the superstructure to ensure they meet safety standards.
        """
        next_steps = "The next stage will involve facade construction, which should follow immediately after the superstructure is completed."

    elif stage == "facade":
        key_insights = f"""
        • Facade: The image has been identified with {accuracy_percentage:.2f}% accuracy as showing the facade stage. 
          The facade construction is progressing well, but it is important to ensure all materials meet the required quality standards.
        """
        recommendations = """
        • Accelerate facade work if necessary to prevent delays to interior construction.
        """
        next_steps = "Prepare for interior work once facade construction is completed."

    elif stage == "interior":
        key_insights = f"""
        • Interior: The image has been identified with {accuracy_percentage:.2f}% accuracy as showing the interior stage.
          Ensure that all installations and fittings are according to the design plan to avoid rework in later stages.
        """
        recommendations = """
        • Regularly check the progress of interior work and address any bottlenecks related to materials or labor.
        """
        next_steps = "Final finishing and quality assurance checks will follow the interior stage."

    else:
        key_insights = "Invalid construction stage provided."
        recommendations = ""
        next_steps = ""

    # Generate the full report using the template
    report_template = f"""
    Construction Progress Report
    Project Name: {project_name}
    Current Stage: {stage.capitalize()}
    Accuracy: {accuracy_percentage:.2f}% matching the stage '{stage.capitalize()}'

    Overview: Based on the image provided, the construction is in the {stage.lower()} stage, identified with {accuracy_percentage:.2f}% accuracy.
    
    Key Insights:
    {key_insights}

    Recommendations:
    {recommendations}

    Next Steps: {next_steps}
    """

    return report_template

# Generate PDF from the report
def generate_pdf(report_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text = c.beginText(40, height - 40)
    text.setFont("Helvetica", 12)

    for line in report_text.split('\n'):
        text.textLine(line)
    
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    
    return buffer

# Function to send the report via email
def send_email(to_email, subject, body, pdf_buffer, file_name):
    from_email = "atlaslily0987@gmail.com"  # Replace with your email
    password = "lzva cnvi ewpm piht"  # Use an app-specific password if using Gmail

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Attach the PDF file
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(pdf_buffer.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename={file_name}")
    msg.attach(part)

    # Set up the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

# Streamlit app UI
st.title('CUBES - Construction Image Classification')
st.write("Upload an image to classify the current stage of construction using a trained CNN model.")

# Get project name input from the user
project_name = st.text_input("Enter the project name:", value="Project X")
user_email = st.text_input("Enter your email address to receive the report:")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Initialize session state for prediction and report
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'stage' not in st.session_state:
    st.session_state.stage = None
if 'report' not in st.session_state:
    st.session_state.report = None

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for model input
    processed_img = preprocess_image(img)

    # Classify the image when the button is clicked
    if st.button('Classify Image'):
        st.write("Classifying...")
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        # Store the classification results in session state
        st.session_state.stage = class_labels[predicted_class]
        st.session_state.confidence = confidence
        st.session_state.prediction = prediction  # Store the full prediction

        # Generate and store the report in session state
        st.session_state.report = generate_construction_report(project_name, st.session_state.stage, st.session_state.confidence)

# Display the classification and report if they exist
if st.session_state.stage:
    st.write(f"Prediction: {st.session_state.stage.capitalize()}")
    st.write(f"Confidence: {st.session_state.confidence:.2f}%")
    
    # Optional: Visualize prediction confidence for all classes
    st.bar_chart(st.session_state.prediction[0])

    # Display the report
    st.write("### Generated Construction Report:")
    st.text(st.session_state.report)

    # Generate the PDF buffer
    pdf_buffer = generate_pdf(st.session_state.report)

    # Add download button for the PDF report
    st.download_button(
        label="Download Report as PDF",
        data=pdf_buffer,
        file_name=f"{project_name}_construction_report.pdf",
        mime="application/pdf"
    )

    # Send the report via email when the button is clicked
    if st.button("Send Report via Email"):
        if user_email:
            send_email(user_email, f"{project_name} Construction Report", "Please find attached your construction report.", pdf_buffer, f"{project_name}_construction_report.pdf")
            st.success(f"Email sent to {user_email}!")
        else:
            st.error("Please enter a valid email address.")