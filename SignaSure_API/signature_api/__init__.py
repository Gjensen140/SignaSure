from flask import Flask
from signature_api.database import create_db

def create_app():
    """ Factory function to create the Flask app """
    app = Flask(__name__)

    # Creates the database (only runs if not initialized)
    create_db()

    # Configuration settings (can be expanded later)
    app.config["SECRET_KEY"] = "supersecretkey"

    # Import and register routes
    from SignaSure_API.sig_api import main
    app.register_blueprint(main)

    return app
