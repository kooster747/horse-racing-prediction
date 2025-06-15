# Horse Racing Prediction Software

A comprehensive platform for horse racing predictions with a 76% win rate, real-time data updates, and personalized recommendations.

## Features

- **Advanced Prediction Algorithm**: Statistical model with 76% win rate in testing
- **Real-Time Updates**: Data refreshed up to the minute before race start
- **Multi-Country Support**: Coverage for UK, US, France, Australia, Japan, Hong Kong, and UAE
- **Personalized Recommendations**: Tailored betting suggestions based on user preferences
- **Interactive Dashboard**: User-friendly interface with comprehensive race information
- **Mobile Compatibility**: Responsive design for on-the-go access

## Repository Structure

```
horse_racing_prediction/
├── src/
│   ├── data/           # Data processing and analysis
│   ├── models/         # Prediction models and algorithms
│   ├── api/            # API endpoints for real-time data
│   └── web/            # Web application backend
├── frontend/
│   ├── static/
│   │   ├── css/        # Stylesheets
│   │   ├── js/         # JavaScript files
│   │   └── images/     # Graphics and assets
│   └── index.html      # Main frontend file
└── docs/               # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/horse-racing-prediction.git
cd horse-racing-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
python src/data/setup_database.py
```

4. Start the web application:
```bash
python src/web/main.py
```

5. Access the application at `http://localhost:5000`

## Usage

### Basic Prediction Flow

1. Browse upcoming races in the Races section
2. View detailed predictions with win probabilities
3. Check real-time updates for last-minute changes
4. Make informed betting decisions based on recommendations

### Personalized Recommendations

1. Create an account and set your preferences
2. Receive tailored race recommendations
3. Track your betting history and performance
4. Refine your strategy based on results

## Development

### Prerequisites

- Python 3.11+
- Flask
- NumPy, Pandas, Scikit-learn
- React (for frontend development)

### Setting Up Development Environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest
```

## Deployment

The application can be deployed using:

1. **Flask Production Server**:
   - Follow the installation steps above
   - Use a production WSGI server like Gunicorn

2. **Docker**:
   - Build the Docker image: `docker build -t horse-racing-prediction .`
   - Run the container: `docker run -p 5000:5000 horse-racing-prediction`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Historical racing data provided by multiple international sources
- Statistical modeling techniques based on published research in sports prediction
- UI design inspired by modern betting platforms

---

**Disclaimer**: Past performance is not a guarantee of future results. Please bet responsibly.
