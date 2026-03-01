⚙️ Installation & Setup
Clone the repository:

Create a virtual environment:

Install dependencies:

Run the Application:

📊 How the Forecasting Works
The system performs a competitive "back-test" on historical data. It hides the most recent 20% of sales data, trains the models on the remaining 80%, and measures the Mean Absolute Error (MAE). The model with the highest accuracy is then used to project sales into the future (7 or 30-day horizons).

📝 Configuration
To update the "Detailed Explanation" in the UI without changing code, edit the config/ui_content.json file. Ensure the file is saved with UTF-8 encoding to support emojis and special characters.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
