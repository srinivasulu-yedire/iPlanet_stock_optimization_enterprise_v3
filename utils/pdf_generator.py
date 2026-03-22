from fpdf import FPDF
from datetime import datetime
from utils.logger import log

def generate_strategy_pdf(product_name, model_name, inventory_metrics, ai_summary):
    """
    Generates a professional PDF report containing the inventory metrics and AI strategy.
    """
    log(f"=== GENERATING PDF FOR {product_name} ===")
    
    class PDF(FPDF):
        def header(self):
            # Header
            self.set_font("helvetica", "B", 15)
            self.cell(0, 10, "iPlanet Enterprise - Inventory Strategy Report", border=False, ln=True, align="C")
            self.set_font("helvetica", "I", 10)
            self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", border=False, ln=True, align="C")
            self.ln(5)

        def footer(self):
            # Footer
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # Initialize PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Section 1: Product Overview
    pdf.set_font("helvetica", "B", 12)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(0, 10, f" Target Product: {product_name}", fill=True, ln=True)
    pdf.ln(5)

    # Section 2: Inventory Directives
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 8, "Calculated Inventory Metrics (Math Engine):", ln=True)
    
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Winning Forecast Algorithm: {model_name}", ln=True)
    pdf.cell(0, 6, f"Economic Order Quantity (EOQ): {inventory_metrics.get('EOQ')} units", ln=True)
    pdf.cell(0, 6, f"Safety Stock: {inventory_metrics.get('Safety Stock')} units", ln=True)
    pdf.cell(0, 6, f"Reorder Point (ROP): {inventory_metrics.get('Reorder Point')} units", ln=True)
    pdf.cell(0, 6, f"Total Projected Annual Cost: ${inventory_metrics.get('Total Annual Cost'):,.2f}", ln=True)
    pdf.ln(10)

# Section 3: AI Strategy (Cleaned up for PDF)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, " Executive Strategy & Action Plan (AI-Generated):", fill=True, ln=True)
    pdf.ln(5)
    
    # Clean the markdown characters from the GenAI output
    clean_summary = ai_summary.replace('**', '').replace('###', '\n>>').replace('* ', '- ')
    
    # CRITICAL FIX: Safely replace emojis and force the text into a format Helvetica understands
    clean_summary = clean_summary.replace('⚠️', '[WARNING]')
    clean_summary = clean_summary.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, clean_summary)

    # Output as standard bytes for Streamlit
    return bytes(pdf.output())
