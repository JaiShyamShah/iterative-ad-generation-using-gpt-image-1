import streamlit as st
import base64
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key with validation
api_key_openai = os.getenv('OPENAI_API_KEY')
if not api_key_openai:
    st.error("OPENAI_API_KEY not found in environment variables. Please create a .env file with your OpenAI API key.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key_openai)

def generate_ad_concept(brand_info, target_audience, marketing_goal):
    """Generate Facebook ad concept using GPT-4"""
    st.info("Generating initial ad concept...")
    
    prompt = f"""
    Create a Facebook ad concept based on:
    - Brand: {brand_info}
    - Audience: {target_audience}
    - Goal: {marketing_goal}
    
    Return JSON with these fields:
    - headline: Catchy headline (5-7 words)
    - primary_text: Main ad copy (1-2 sentences)
    - description: Additional context (optional)
    - cta: Call-to-action (e.g., "Shop Now")
    - image_edit_instructions: Detailed instructions for image editing
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional ad copywriter. Return only valid JSON with all required fields."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate all required fields are present
        required_fields = ['headline', 'primary_text', 'cta', 'image_edit_instructions']
        if all(field in result for field in required_fields):
            return result
        else:
            st.error(f"Missing required fields in response: {result}")
            return None
            
    except Exception as e:
        st.error(f"Failed to generate concept: {str(e)}")
        return None

def generate_initial_image(prompt):
    """Generate initial image using GPT-Image-1"""
    st.info("Generating initial image...")
    
    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        # Get the base64 encoded image directly
        image_b64 = response.data[0].b64_json
        return f"data:image/png;base64,{image_b64}"
            
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        return None

def edit_image_with_prompt(base_image, edit_instructions):
    """Edit existing image using GPT-Image-1"""
    st.info("Editing image based on feedback...")
    
    try:
        # Decode the base64 image data
        image_data = base64.b64decode(base_image.split(",")[1])
        
        # Save temporarily to file (OpenAI API requires a file)
        with open("temp_image.png", "wb") as f:
            f.write(image_data)
        
        # Log the edit instructions
        st.write("### Edit Instructions Sent to API")
        st.write(edit_instructions)
        
        # Open file for API
        with open("temp_image.png", "rb") as img_file:
            # Ensure edit_instructions is a string
            if not isinstance(edit_instructions, str):
                edit_instructions = str(edit_instructions)
                
            response = client.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=edit_instructions,
                n=1,
                size="1024x1024"
            )
        
        # Get the base64 encoded image directly
        image_b64 = response.data[0].b64_json
        
        # Clean up temp file
        if os.path.exists("temp_image.png"):
            os.remove("temp_image.png")
            
        return f"data:image/png;base64,{image_b64}"
            
    except openai.error.InvalidRequestError as e:
        if e.error and e.error.get('code') == 'moderation_blocked':
            st.error("The edit request was blocked by the safety system. Please review the edit instructions above and ensure they are appropriate. Consider simplifying or modifying the instructions.")
        else:
            st.error(f"Image editing error: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists("temp_image.png"):
            os.remove("temp_image.png")
        return None
    except Exception as e:
        st.error(f"Image editing error: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists("temp_image.png"):
            os.remove("temp_image.png")
        return None

def analyze_and_improve(image_b64, ad_concept, iteration):
    """Analyze ad and suggest editing improvements"""
    st.info(f"Analyzing iteration {iteration}...")
    
    critique_prompt = f"""
    Analyze this Facebook ad (iteration {iteration}) and suggest editing improvements:
    
    Current Ad:
    - Headline: {ad_concept['headline']}
    - Primary Text: {ad_concept['primary_text']}
    - CTA: {ad_concept['cta']}
    
    Provide specific feedback on:
    1. Visual elements that need modification
    2. Composition adjustments
    3. Color scheme improvements
    4. Element positioning
    
    When suggesting edit instructions, be as specific as possible and ensure they are safe, professional, and suitable for all audiences. Avoid any requests that could be interpreted as explicit, violent, or inappropriate. Focus on aesthetic improvements such as color adjustments, composition changes, and element positioning.
    
    For example, instead of saying 'make it better', say 'increase the brightness of the background' or 'add a soft shadow to the text'.
    
    Return JSON with:
    - critique: Your analysis
    - edit_instructions: Detailed editing instructions for GPT-Image-1
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert image editor. When providing edit instructions, ensure they are safe, professional, and suitable for all audiences. Avoid any requests that could be interpreted as explicit, violent, or inappropriate. Focus on visual improvements such as color adjustments, composition changes, and element positioning. Return only valid JSON with 'critique' and 'edit_instructions' fields."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_b64,
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": critique_prompt
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate response contains required fields
        if 'critique' in result and 'edit_instructions' in result:
            return result
        else:
            st.error(f"Missing required fields in critique: {result}")
            return None
            
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

def main():
    st.title("🖼️ AI Image Editing Studio")
    st.subheader("Iterative image editing with GPT-Image-1")
    
    # Initialize session state
    if 'iterations' not in st.session_state:
        st.session_state.iterations = []
    if 'current_iteration' not in st.session_state:
        st.session_state.current_iteration = 0
    if 'max_iterations' not in st.session_state:
        st.session_state.max_iterations = 3
    if 'ad_concept' not in st.session_state:
        st.session_state.ad_concept = None
    
    # Configuration sidebar
    with st.sidebar:
        st.header("Settings")
        st.session_state.max_iterations = st.slider("Number of edits", 1, 10, 3)
    
    # Input form
    with st.form("ad_input_form"):
        st.write("### Campaign Details")
        brand_info = st.text_area("Brand Information", 
                                "Brand: EcoWear\nProducts: Sustainable activewear\nUSP: Eco-friendly materials")
        target_audience = st.text_area("Target Audience", 
                                     "Age 25-40, eco-conscious, fitness enthusiasts")
        marketing_goal = st.text_area("Marketing Goal", 
                                    "Launch new summer collection, drive website traffic")
        
        if st.form_submit_button("Start Editing Process"):
            st.session_state.iterations = []
            st.session_state.current_iteration = 0
            st.session_state.ad_concept = generate_ad_concept(brand_info, target_audience, marketing_goal)
            
            if st.session_state.ad_concept:
                # Generate initial image using proper generation endpoint
                initial_image = generate_initial_image(
                    st.session_state.ad_concept['image_edit_instructions']
                )
                
                if initial_image:
                    st.session_state.iterations.append({
                        'iteration': 0,
                        'image': initial_image,
                        'edit_instructions': st.session_state.ad_concept['image_edit_instructions']
                    })
                    st.rerun()
                else:
                    st.error("Failed to generate initial image")


    # Display current status
    if st.session_state.ad_concept:
        st.write("### Current Editing Instructions")
        st.json(st.session_state.ad_concept)
    
    # Generate editing iterations
    if st.session_state.ad_concept and st.session_state.current_iteration < st.session_state.max_iterations:
        if st.button(f"Apply Edit {st.session_state.current_iteration + 1}"):
            if not st.session_state.iterations:
                st.error("No initial image found - please start the editing process first")
                return
                
            with st.spinner(f"Applying edit {st.session_state.current_iteration + 1}..."):
                last_image = st.session_state.iterations[-1]['image']
                
                # Get analysis and edit instructions
                analysis = analyze_and_improve(
                    last_image,
                    st.session_state.ad_concept,
                    st.session_state.current_iteration
                )
                
                if not analysis:
                    st.error("Failed to generate edit instructions")
                    return
                
                # Apply edit - Use the string value directly from analysis
                edited_image = edit_image_with_prompt(
                    last_image,
                    analysis['edit_instructions']  # This should be a string
                )
                
                if edited_image:
                    st.session_state.iterations.append({
                        'iteration': st.session_state.current_iteration + 1,
                        'image': edited_image,
                        'edit_instructions': analysis['edit_instructions'],
                        'critique': analysis['critique']
                    })
                    st.session_state.current_iteration += 1
                    st.rerun()
                else:
                    st.error("Failed to apply edit")

    # Display all iterations
    if st.session_state.iterations:
        st.write("## Editing History")
        
        for i, iteration in enumerate(st.session_state.iterations):
            with st.expander(f"Edit {i}", expanded=i == len(st.session_state.iterations)-1):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(iteration['image'], use_column_width=True)
                    st.caption(f"Edit instructions: {iteration.get('edit_instructions', 'No instructions')}")
                
                with col2:
                    if 'critique' in iteration:
                        st.write("### Feedback")
                        st.write(iteration['critique'])

    # Final result
    if (st.session_state.iterations and 
        st.session_state.current_iteration >= st.session_state.max_iterations):
        st.success("🎉 Editing process complete!")
        
        if st.session_state.iterations:
            final_image = st.session_state.iterations[-1]['image']
            st.download_button(
                label="Download Final Image",
                data=base64.b64decode(final_image.split(",")[1]),
                file_name="edited_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()