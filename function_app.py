import azure.functions as func
import logging
from Services.app_detect_image import process_image

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="analzeImage", methods=["POST"])
def analzeImage(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Get the uploaded image file
        image_file = req.files.get('image')
        
        if not image_file:
            return func.HttpResponse(
                "Please provide an image file in the request body with key 'image'.",
                status_code=400
            )
        
        # Read the image data
        image_data = image_file.read()
        
        logging.info(f"Received image file: {image_file.filename}, size: {len(image_data)} bytes")
        
        # Call the process_image function
        result_json = process_image(image_data)
        
        return func.HttpResponse(
            result_json,
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return func.HttpResponse(
            f"Error processing image: {str(e)}",
            status_code=500
        )