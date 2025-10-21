from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import wikipediaapi
from dotenv import load_dotenv
import os
import json
import re
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

class EduGeniusEngine:
    def __init__(self):
        self.initialize_apis()
        
    def initialize_apis(self):
        """Initialize Gemini and Wikipedia APIs"""
        try:
            # Configure Gemini API
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize Wikipedia API
            self.wiki = wikipediaapi.Wikipedia(
                language='en',
                user_agent='EduGenius/2.0 (Educational Content Generator)'
            )
            
        except Exception as e:
            print(f"Error initializing APIs: {str(e)}")
            raise
    
    def parse_learning_request(self, user_prompt):
        """Parse natural language learning request to extract key information"""
        try:
            parsing_prompt = f"""
            Analyze this learning request and extract key information in JSON format:
            
            User Request: "{user_prompt}"
            
            Extract the following information:
            1. main_topic: The primary subject/topic to learn about
            2. subject_domain: The broader academic field (e.g., Computer Science, Biology, History)
            3. specific_focus: Any specific aspects or subtopics mentioned
            4. learning_intent: What the user wants to achieve (learn, understand, master, etc.)
            5. complexity_level: Inferred level (beginner, intermediate, advanced) based on language used
            6. keywords: List of important terms for Wikipedia search
            
            Return ONLY a valid JSON object with these fields. Do not include any other text.
            
            Example output:
            {{
                "main_topic": "System Calls",
                "subject_domain": "Computer Science - Operating Systems", 
                "specific_focus": "System calls in operating systems",
                "learning_intent": "comprehensive understanding",
                "complexity_level": "intermediate",
                "keywords": ["system calls", "operating systems", "kernel", "user space"]
            }}
            """
            
            response = self.model.generate_content(parsing_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                parsed_info = json.loads(json_match.group())
                return parsed_info, None
            else:
                # Fallback parsing
                return self.fallback_parse(user_prompt), None
                
        except Exception as e:
            return None, f"Error parsing request: {str(e)}"
    
    def fallback_parse(self, user_prompt):
        """Fallback parsing method if JSON extraction fails"""
        # Simple keyword extraction
        common_domains = {
            'computer science': ['programming', 'algorithm', 'data structure', 'system', 'software'],
            'biology': ['cell', 'organism', 'dna', 'protein', 'evolution'],
            'physics': ['quantum', 'energy', 'force', 'particle', 'wave'],
            'chemistry': ['molecule', 'atom', 'reaction', 'compound', 'element'],
            'mathematics': ['equation', 'theorem', 'calculus', 'algebra', 'geometry']
        }
        
        prompt_lower = user_prompt.lower()
        
        # Try to identify domain
        identified_domain = "General"
        for domain, keywords in common_domains.items():
            if any(keyword in prompt_lower for keyword in keywords):
                identified_domain = domain.title()
                break
        
        return {
            "main_topic": user_prompt.replace("teach me about", "").replace("learn about", "").strip(),
            "subject_domain": identified_domain,
            "specific_focus": user_prompt,
            "learning_intent": "comprehensive understanding",
            "complexity_level": "intermediate",
            "keywords": prompt_lower.split()[:5]
        }
    
    def fetch_wikipedia_context(self, keywords, main_topic):
        """Fetch relevant Wikipedia content based on keywords and topic"""
        try:
            contexts = []
            
            # Try main topic first
            search_terms = [main_topic] + keywords[:3]
            
            for term in search_terms:
                if len(contexts) >= 2:  # Limit to prevent too much content
                    break
                    
                page = self.wiki.page(term)
                
                if page.exists():
                    summary = page.summary
                    if summary and len(summary.strip()) > 100:
                        # Limit summary length
                        if len(summary) > 1500:
                            summary = summary[:1500] + "..."
                        contexts.append({
                            "title": page.title,
                            "content": summary,
                            "url": page.fullurl
                        })
            
            if not contexts:
                return None, "No relevant Wikipedia content found"
            
            return contexts, None
            
        except Exception as e:
            return None, f"Error fetching Wikipedia data: {str(e)}"
    
    def generate_comprehensive_study_material(self, parsed_info, wikipedia_contexts, original_prompt):
        """Generate comprehensive study material using Gemini"""
        try:
            # Construct context from Wikipedia
            wiki_context = ""
            for i, ctx in enumerate(wikipedia_contexts):
                wiki_context += f"\n--- Source {i+1}: {ctx['title']} ---\n{ctx['content']}\n"
            
            study_prompt = f"""
            You are an expert educator and curriculum designer. Create comprehensive, well-structured study material based on the user's learning request.

            **Original User Request:** "{original_prompt}"

            **Parsed Learning Goals:**
            - Topic: {parsed_info['main_topic']}
            - Domain: {parsed_info['subject_domain']}
            - Focus: {parsed_info['specific_focus']}
            - Level: {parsed_info['complexity_level']}

            **Factual Reference Material:**
            {wiki_context}

            **Your Task:**
            Create a comprehensive study guide that is:
            1. **Structured** - Use clear headings and logical flow
            2. **Comprehensive** - Cover all important aspects
            3. **Concise** - Be thorough but not overwhelming
            4. **Effective** - Focus on key concepts that aid understanding
            5. **Practical** - Include examples where relevant

            **Required Structure:**
            
            # {parsed_info['main_topic']} - Complete Study Guide
            
            ## 📋 Overview
            [Brief introduction and why this topic is important]
            
            ## 🎯 Learning Objectives
            [What you'll understand after studying this material]
            
            ## 🔑 Key Concepts
            [Core concepts with clear definitions]
            
            ## 📖 Detailed Explanation
            [In-depth explanation of the topic]
            
            ## 💡 Practical Examples
            [Real-world applications or examples]
            
            ## 🔗 Key Relationships
            [How this topic connects to other concepts]
            
            ## ⚡ Quick Reference
            [Important formulas, commands, or summary points]
            
            ## 🧠 Self-Assessment Questions
            [3-5 questions to test understanding]
            
            ## 📚 Further Study
            [Suggestions for deeper learning]

            Generate content that is pedagogically sound, engaging, and tailored to the {parsed_info['complexity_level']} level.
            Base your content on the provided reference material but expand with educational context as needed.
            """
            
            response = self.model.generate_content(study_prompt)
            
            if not response.text:
                return None, "Failed to generate study material"
            
            return response.text, None
            
        except Exception as e:
            return None, f"Error generating study material: {str(e)}"

# Initialize the engine
edu_engine = EduGeniusEngine()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_study_material():
    """Generate study material based on user prompt"""
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '').strip()
        
        if not user_prompt:
            return jsonify({
                'success': False,
                'error': 'Please provide a learning prompt'
            })
        
        # Step 1: Parse the learning request
        parsed_info, parse_error = edu_engine.parse_learning_request(user_prompt)
        if parse_error:
            return jsonify({
                'success': False,
                'error': parse_error
            })
        
        # Step 2: Fetch Wikipedia context
        wikipedia_contexts, wiki_error = edu_engine.fetch_wikipedia_context(
            parsed_info['keywords'], 
            parsed_info['main_topic']
        )
        
        if wiki_error:
            return jsonify({
                'success': False,
                'error': wiki_error
            })
        
        # Step 3: Generate comprehensive study material
        study_material, content_error = edu_engine.generate_comprehensive_study_material(
            parsed_info, wikipedia_contexts, user_prompt
        )
        
        if content_error:
            return jsonify({
                'success': False,
                'error': content_error
            })
        
        # Return successful response
        return jsonify({
            'success': True,
            'data': {
                'study_material': study_material,
                'parsed_info': parsed_info,
                'sources': [ctx['title'] for ctx in wikipedia_contexts],
                'generated_at': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'EduGenius API',
        'version': '2.0'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8007)
