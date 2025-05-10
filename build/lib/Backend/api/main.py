from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from Backend.models.Diet_rec_model import recommend, output_recommended_recipes
import io
import numpy as np
import torch
from PIL import Image
from Backend.models.segmentation_model import segmentation_model
from Backend.models.segmentation_model import run_inference
from fastapi.middleware.cors import CORSMiddleware
from mmseg.apis import inference_model  # Update this import
import logging



dataset = pd.read_csv('../data/dataset.csv', compression='gzip')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class params(BaseModel):
    n_neighbors:int=5
    return_distance:bool=False

class PredictionIn(BaseModel):
    # Use Field instead of conlist for array validation
    nutrition_input: List[float] = Field(..., min_items=9, max_items=9)
    ingredients: List[str] = []
    params:Optional[params]

class Recipe(BaseModel):
    Name:str
    CookTime:str
    PrepTime:str
    TotalTime:str
    RecipeIngredientParts:list[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict/",response_model=PredictionOut)
def update_item(prediction_input:PredictionIn):
    recommendation_dataframe=recommend(dataset,prediction_input.nutrition_input,prediction_input.ingredients,prediction_input.params.dict())
    output=output_recommended_recipes(recommendation_dataframe)
    if output is None:
        return {"output":None}
    else:
        return {"output":output}
    
class SegmentationResult(BaseModel):
    segmentation_mask: List[List[int]]
    detected_ingredients: List[str]
    ingredient_percentages: dict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.post("/segment_food/", response_model=SegmentationResult)
async def segment_food(file: UploadFile = File(...)):
    try:
        logger.info("Starting food segmentation request")
        
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG and PNG images are supported."
            )
        
        # Read and preprocess the image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Ensure model is loaded
        if segmentation_model is None:
            logger.error("Model not initialized")
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Run inference
        segmentation_mask = run_inference(segmentation_model, image)
        
        if segmentation_mask is None:
            logger.error("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed")
        
        # Calculate detected ingredients and their percentages
        unique_classes = np.unique(segmentation_mask)
        detected_ingredients = []
        ingredient_percentages = {}
    
        CLASS_NAMES = [
            "background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream",
            "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew",
            "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana",
            "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig",
            "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage",
            "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn",
            "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant",
            "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra",
            "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint",
            "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom",
            "oyster mushroom", "white button mushroom", "salad", "other ingredients"
        ]
    
        try:
            for class_id in unique_classes:
                if class_id != 0:  # Exclude background
                    class_name = CLASS_NAMES[class_id]
                    pixel_count = np.sum(segmentation_mask == class_id)
                    percentage = (pixel_count / segmentation_mask.size) * 100
                    detected_ingredients.append(class_name)
                    ingredient_percentages[class_name] = round(percentage, 2)
        except Exception as e:
            logger.error(f"Error processing segmentation results: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing segmentation results")
        
        logger.info("Successfully completed segmentation request")
        return {
            "segmentation_mask": segmentation_mask.tolist(),
            "detected_ingredients": detected_ingredients,
            "ingredient_percentages": ingredient_percentages
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in segment_food: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))