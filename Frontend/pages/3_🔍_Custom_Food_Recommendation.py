import streamlit as st
from Generate_Recommendations import Generator
from ImageFinder.ImageFinder import get_images_links as find_image
import pandas as pd
from streamlit_echarts import st_echarts

st.set_page_config(page_title="Custom Food Recommendation", page_icon="🔍",layout="wide")
nutrition_values=['Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']
if 'generated' not in st.session_state:
    st.session_state.generated = False
    st.session_state.recommendations=None

class Recommendation:
    def __init__(self,nutrition_list,nb_recommendations,ingredient_txt):
        self.nutrition_list=nutrition_list
        self.nb_recommendations=nb_recommendations
        self.ingredient_txt=ingredient_txt
        pass
    def generate(self,):
        params = {'n_neighbors': self.nb_recommendations, 'return_distance': False}
        ingredients = self.ingredient_txt.split(';')
        generator = Generator(self.nutrition_list, ingredients, params)
        result = generator.generate()

        # Debug output to inspect structure

        # Safely extract recommendations
        if isinstance(result, dict) and 'output' in result:
            recommendations = result['output']
        else:
            st.error("Invalid response format from recommendation generator.")
            recommendations = []

        # Only add image links if recommendations are a list of dicts
        if recommendations and isinstance(recommendations, list):
            for recipe in recommendations:
                if isinstance(recipe, dict) and 'Name' in recipe:
                    recipe['image_link'] = find_image(recipe['Name'])

        return recommendations


class Display:
    def __init__(self):
        self.nutrition_values=nutrition_values

    def display_recommendation(self,recommendations):
        st.subheader('Recommended recipes:')
        if recommendations!=None and isinstance(recommendations, list):
            # Ensure we have at least 1 row
            rows = max(1, len(recommendations)//5)
            for column,row in zip(st.columns(5),range(5)):
                with column:
                    # Get slice of recommendations for this column
                    column_recipes = recommendations[rows*row:rows*(row+1)] if row < len(recommendations)//rows else []
                    for recipe in column_recipes:
                        # Ensure recipe is a dictionary before accessing keys
                        if not isinstance(recipe, dict):
                            st.warning(f"Skipping invalid recipe format: {type(recipe)}")
                            continue
                            
                        # Safely access recipe data
                        recipe_name = recipe.get('Name', 'Unknown Recipe')
                        expander = st.expander(recipe_name)
                        recipe_link = recipe.get('image_link', '')
                        recipe_img=f'<div><center><img src={recipe_link} alt={recipe_name}></center></div>'     
                        
                        # Safely create nutrition dataframe
                        try:
                            nutritions_df = pd.DataFrame({value:[recipe.get(value, 0)] for value in nutrition_values})
                        except Exception as e:
                            st.error(f"Error creating nutrition dataframe: {e}")
                            nutritions_df = pd.DataFrame()
                        
                        expander.markdown(recipe_img, unsafe_allow_html=True)  
                        expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values (g):</h5>', unsafe_allow_html=True)                   
                        expander.dataframe(nutritions_df)
                        
                        # Safely display ingredients
                        expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Ingredients:</h5>', unsafe_allow_html=True)
                        ingredients = recipe.get('RecipeIngredientParts', [])
                        if isinstance(ingredients, list) and ingredients:
                            for ingredient in ingredients:
                                expander.markdown(f"""
                                        - {ingredient}
                                """)
                        else:
                            expander.info("No ingredient information available")
                            
                        # Safely display instructions
                        expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Recipe Instructions:</h5>', unsafe_allow_html=True)    
                        instructions = recipe.get('RecipeInstructions', [])
                        if isinstance(instructions, list) and instructions:
                            for instruction in instructions:
                                expander.markdown(f"""
                                        - {instruction}
                                """)
                        else:
                            expander.info("No instruction information available")
                            
                        # Safely display cooking times
                        expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Cooking and Preparation Time:</h5>', unsafe_allow_html=True)   
                        expander.markdown(f"""
                                - Cook Time       : {recipe.get('CookTime', 'N/A')}min
                                - Preparation Time: {recipe.get('PrepTime', 'N/A')}min
                                - Total Time      : {recipe.get('TotalTime', 'N/A')}min
                            """)
        else:
            st.info('Couldn\'t find any recipes with the specified ingredients', icon="🙁")
    def display_overview(self,recommendations):
        if recommendations is None or not isinstance(recommendations, list) or not recommendations:
            st.info('No recommendations available to display overview.')
            return
            
        # Filter out non-dictionary recipes and those without a Name
        valid_recipes = [r for r in recommendations if isinstance(r, dict) and 'Name' in r]
        
        if not valid_recipes:
            st.warning('No valid recipes found to display overview.')
            return
            
        st.subheader('Overview:')
        col1,col2,col3=st.columns(3)
        
        # Safely extract recipe names
        recipe_names = [recipe.get('Name', f'Recipe {i}') for i, recipe in enumerate(valid_recipes)]
        
        with col2:
            selected_recipe_name = st.selectbox('Select a recipe', recipe_names)
        
        st.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values:</h5>', unsafe_allow_html=True)
        
        # Find the selected recipe
        selected_recipe = None
        for recipe in valid_recipes:
            if recipe.get('Name') == selected_recipe_name:
                selected_recipe = recipe
                break
                
        if not selected_recipe:
            st.error('Selected recipe not found.')
            return
            
        # Safely create chart data
        chart_data = []
        for nutrition_value in self.nutrition_values:
            try:
                value = selected_recipe.get(nutrition_value, 0)
                chart_data.append({"value": value, "name": nutrition_value})
            except Exception as e:
                st.error(f"Error processing nutrition value {nutrition_value}: {e}")
                
        options = {
            "title": {"text": "Nutrition values", "subtext": f"{selected_recipe_name}", "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left",},
            "series": [
                {
                    "name": "Nutrition values",
                    "type": "pie",
                    "radius": "50%",
                    "data": chart_data,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ],
        }
        
        try:
            st_echarts(options=options, height="600px",)
            st.caption('You can select/deselect an item (nutrition value) from the legend.')
        except Exception as e:
            st.error(f"Error displaying chart: {e}")

title="<h1 style='text-align: center;'>Custom Food Recommendation</h1>"
st.markdown(title, unsafe_allow_html=True)


display=Display()

with st.form("recommendation_form"):
    st.header('Nutritional values:')
    Calories = st.slider('Calories', 0, 2000, 500)
    FatContent = st.slider('FatContent', 0, 100, 50)
    SaturatedFatContent = st.slider('SaturatedFatContent', 0, 13, 0)
    CholesterolContent = st.slider('CholesterolContent', 0, 300, 0)
    SodiumContent = st.slider('SodiumContent', 0, 2300, 400)
    CarbohydrateContent = st.slider('CarbohydrateContent', 0, 325, 100)
    FiberContent = st.slider('FiberContent', 0, 50, 10)
    SugarContent = st.slider('SugarContent', 0, 40, 10)
    ProteinContent = st.slider('ProteinContent', 0, 40, 10)
    nutritions_values_list=[Calories,FatContent,SaturatedFatContent,CholesterolContent,SodiumContent,CarbohydrateContent,FiberContent,SugarContent,ProteinContent]
    st.header('Recommendation options (OPTIONAL):')
    nb_recommendations = st.slider('Number of recommendations', 5, 20,step=5)
    ingredient_txt=st.text_input('Specify ingredients to include in the recommendations separated by ";" :',placeholder='Ingredient1;Ingredient2;...')
    st.caption('Example: Milk;eggs;butter;chicken...')
    generated = st.form_submit_button("Generate")
if generated:
    with st.spinner('Generating recommendations...'): 
        recommendation=Recommendation(nutritions_values_list,nb_recommendations,ingredient_txt)
        recommendations=recommendation.generate()
        st.session_state.recommendations=recommendations
    st.session_state.generated=True 

if st.session_state.generated:
    with st.container():
        display.display_recommendation(st.session_state.recommendations)
    with st.container():
        display.display_overview(st.session_state.recommendations)