from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Report paths
report_path = "results/final_report.pdf"
confusion_matrix_path = "results/confusion_matrix.png"

# Create document
doc = SimpleDocTemplate(report_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("<b>AI-Powered Image Classification System</b>", styles['Title']))
story.append(Spacer(1, 12))

# Model Info
story.append(Paragraph("<b>1. Project Overview</b>", styles['Heading2']))
story.append(Paragraph("This project uses a Convolutional Neural Network (CNN) model to classify images into five categories — Birds, Cats, Dogs, Fruits, and Tiger/Lion. The model was trained using TensorFlow and evaluated on a separate test dataset.", styles['BodyText']))
story.append(Spacer(1, 12))

# Dataset Info
story.append(Paragraph("<b>2. Dataset Details</b>", styles['Heading2']))
story.append(Paragraph("The dataset was divided into three parts: Train, Validation, and Test sets. Each category contains images representing its class. Images were resized and normalized before training.", styles['BodyText']))
story.append(Spacer(1, 12))

# Results
story.append(Paragraph("<b>3. Evaluation Results</b>", styles['Heading2']))
data = [
    ["Metric", "Value"],
    ["Test Accuracy", "73.32%"],
    ["Test Loss", "1.4956"],
    ["Total Classes", "5 (birds, cats, dogs, fruits, tiger_lion)"]
]
table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.grey),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0,0), (-1,0), 12),
    ('BACKGROUND',(0,1),(-1,-1),colors.beige),
]))
story.append(table)
story.append(Spacer(1, 20))

# Confusion Matrix Image
story.append(Paragraph("<b>4. Confusion Matrix</b>", styles['Heading2']))
story.append(Paragraph("The confusion matrix below shows how well the model distinguished between classes.", styles['BodyText']))
story.append(Spacer(1, 10))
story.append(Image(confusion_matrix_path, width=400, height=300))
story.append(Spacer(1, 20))

# Conclusion
story.append(Paragraph("<b>5. Conclusion</b>", styles['Heading2']))
story.append(Paragraph("The trained CNN model achieved a test accuracy of approximately 73%, showing strong performance in classifying cats and dogs, with room for improvement in rare categories such as Tiger/Lion. Future improvements could include more data augmentation and model fine-tuning.", styles['BodyText']))

# Build PDF
doc.build(story)
print(f"✅ Final Report Generated: {report_path}")
