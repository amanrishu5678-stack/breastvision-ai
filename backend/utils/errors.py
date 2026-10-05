from flask import jsonify


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def error_response(code: str, message: str, status: int):
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status
