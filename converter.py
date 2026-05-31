from fpdf import FPDF

def text_to_pdf(input_txt_file, output_pdf_file):
    # Initialize the PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set font (Helvetica is standard and built-in)
    pdf.set_font("Helvetica", size=12)
    
    # Open and read the text file
    with open(input_txt_file, "r", encoding="utf-8") as file:
        for line in file:
            # Removed 'ln=True' as modern fpdf2 handles line breaks automatically
            pdf.multi_cell(0, 8, txt=line)
            
    # Save the generated PDF
    pdf.output(output_pdf_file)
    print(f"Success! '{input_txt_file}' has been converted to '{output_pdf_file}'.")

# --- Example Usage ---
# Replace these names with your actual file names
my_text_file = "galaxy.txt"
my_output_pdf = "galaxy.pdf"

text_to_pdf(my_text_file, my_output_pdf)