import os
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. Install with: pip install openai")
    exit(1)

class SimpleRAGAgent:
    """
    Simple but Effective RAG Agent
    Uses smart rule-based matching + OpenAI for maximum reliability
    """
    
    def __init__(self, openai_api_key: str = None, confidence_threshold: float = 0.55):
        """Initialize the Simple RAG Agent with optimized threshold"""
        self.confidence_threshold = confidence_threshold
        
        # Initialize OpenAI
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                raise ValueError("OpenAI API key required")
        
        print("✅ OpenAI client initialized")
        
        # Load FAQ knowledge base
        self.faq_data = self._load_faq_data()
        print(f"✅ Loaded {len(self.faq_data)} FAQ items")
        
        # Conversation history
        self.conversation_history = []
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant for BlueBuck AI.

Your job is to answer questions about BlueBuck AI based on the provided FAQ context.

Rules:
1. Use ONLY the provided FAQ context to answer questions
2. If the context contains the answer, provide a detailed and helpful response
3. You can rephrase and elaborate on the FAQ information to better match the user's question
4. If the context doesn't contain enough information, say: "I'm not sure about that. Want me to escalate it?"
5. Be conversational, professional, and helpful
6. Include specific details like prices, features, and benefits when available

FAQ Context:"""
    
    def _load_faq_data(self) -> List[Dict]:
        """Load FAQ data with comprehensive keywords"""
        return [
            {
                "id": "faq_1",
                "question": "What is BlueBuck AI?",
                "answer": "BlueBuck AI is an advanced artificial intelligence platform designed to help businesses automate processes, analyze data, and make intelligent decisions using cutting-edge machine learning algorithms.",
                "category": "general",
                "keywords": ["what", "is", "bluebuck", "ai", "platform", "about", "artificial", "intelligence", "automation", "overview", "definition", "describe"]
            },
            {
                "id": "faq_2", 
                "question": "What are the pricing plans for BlueBuck AI?",
                "answer": "BlueBuck AI offers three pricing tiers: Starter Plan ($99/month) for small businesses with basic automation features, Professional Plan ($299/month) for growing companies with advanced features and custom integrations, and Enterprise Plan (custom pricing) for large organizations with unlimited usage and dedicated support.",
                "category": "pricing",
                "keywords": ["pricing", "price", "cost", "plans", "plan", "starter", "professional", "enterprise", "monthly", "much", "expensive", "cheap", "affordable", "tier", "subscription"]
            },
            {
                "id": "faq_3",
                "question": "Is there a free trial available?",
                "answer": "Yes! BlueBuck AI offers a comprehensive 14-day free trial for all new users. You can access most platform features during the trial period with no credit card required. This gives you the opportunity to explore our automation tools and see how they can benefit your business.",
                "category": "trial",
                "keywords": ["free", "trial", "demo", "test", "try", "14-day", "credit", "card", "evaluation", "sample", "preview"]
            },
            {
                "id": "faq_4",
                "question": "What integrations does BlueBuck AI support?",
                "answer": "BlueBuck AI integrates seamlessly with popular business tools including Salesforce, HubSpot, Slack, Microsoft Office 365, Google Workspace, MySQL, PostgreSQL, AWS, Azure, and many others through our robust REST API. We also provide custom integration development for Enterprise clients.",
                "category": "integrations",
                "keywords": ["integration", "integrate", "api", "connect", "salesforce", "hubspot", "slack", "office", "google", "microsoft", "aws", "azure", "compatible", "support", "tools"]
            },
            {
                "id": "faq_5",
                "question": "How secure is my data with BlueBuck AI?",
                "answer": "BlueBuck AI prioritizes data security with enterprise-grade measures including end-to-end encryption, SOC 2 Type II compliance, regular third-party security audits, multi-factor authentication, and secure data centers with geographic redundancy. Your data is never shared with third parties and you maintain full ownership.",
                "category": "security",
                "keywords": ["security", "secure", "data", "encryption", "privacy", "protection", "safe", "soc2", "compliance", "audit", "confidential", "gdpr"]
            },
            {
                "id": "faq_6",
                "question": "What customer support does BlueBuck AI provide?",
                "answer": "BlueBuck AI provides comprehensive 24/7 customer support through live chat, email, and phone for all paid plans. Professional and Enterprise plans include dedicated account managers, priority support with faster response times, and custom onboarding sessions. Our average response time is under 2 hours.",
                "category": "support",
                "keywords": ["support", "help", "customer", "service", "assistance", "24/7", "chat", "email", "phone", "contact", "response", "manager", "onboarding"]
            },
            {
                "id": "faq_7",
                "question": "Can I customize BlueBuck AI models?",
                "answer": "Absolutely! With Professional and Enterprise plans, you can train custom AI models using your own data, create custom workflows, configure automation rules, and work with our team to build specialized features. Enterprise clients get dedicated development resources for complex customizations.",
                "category": "customization",
                "keywords": ["customize", "custom", "models", "training", "configure", "personalize", "tailor", "specific", "workflow", "automation", "rules"]
            },
            {
                "id": "faq_8",
                "question": "What industries does BlueBuck AI serve?",
                "answer": "BlueBuck AI serves diverse industries including e-commerce and retail (inventory management, customer service), financial services (fraud detection, risk assessment), healthcare (patient data analysis, appointment scheduling), manufacturing (predictive maintenance, quality control), technology (automated testing, code review), and professional services (document processing, client management).",
                "category": "industries",
                "keywords": ["industries", "industry", "sectors", "ecommerce", "retail", "financial", "healthcare", "manufacturing", "technology", "professional", "business", "serve", "work"]
            },
            {
                "id": "faq_9",
                "question": "How do I get started with BlueBuck AI?",
                "answer": "Getting started is simple: 1) Sign up for your free 14-day trial at bluebuck.ai, 2) Complete our 10-minute onboarding process, 3) Connect your existing tools and data sources, 4) Explore pre-built templates or create custom workflows, 5) Access our tutorials and documentation. Most users are productive within their first day!",
                "category": "getting_started",
                "keywords": ["start", "getting", "started", "begin", "signup", "onboarding", "setup", "tutorial", "guide", "first", "initial"]
            }
        ]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Method 1: Sequence matching
        seq_sim = SequenceMatcher(None, text1_lower, text2_lower).ratio()
        
        # Method 2: Word overlap
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
        else:
            word_overlap = 0
        
        # Method 3: Substring matching
        substring_matches = 0
        words1_list = text1_lower.split()
        for word in words1_list:
            if len(word) > 3 and word in text2_lower:
                substring_matches += 1
        substring_sim = substring_matches / len(words1_list) if words1_list else 0
        
        # Combined similarity
        return (seq_sim * 0.4) + (word_overlap * 0.4) + (substring_sim * 0.2)
    
    def _calculate_keyword_score(self, query: str, keywords: List[str]) -> float:
        """Calculate keyword matching score"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Direct keyword matches
        direct_matches = 0
        for keyword in keywords:
            if keyword.lower() in query_lower:
                direct_matches += 1
        
        # Word-level matches
        keyword_words = set()
        for keyword in keywords:
            keyword_words.update(keyword.lower().split())
        
        word_matches = len(query_words & keyword_words)
        
        # Calculate score
        if keywords:
            direct_score = direct_matches / len(keywords)
            word_score = word_matches / len(keyword_words) if keyword_words else 0
            return (direct_score * 0.7) + (word_score * 0.3)
        
        return 0
    
    def _smart_retrieval(self, query: str, top_k: int = 3) -> List[Dict]:
        """Smart retrieval using multiple scoring methods"""
        
        scored_items = []
        
        for item in self.faq_data:
            # Calculate different similarity scores
            question_sim = self._calculate_text_similarity(query, item["question"])
            answer_sim = self._calculate_text_similarity(query, item["answer"]) * 0.3
            keyword_score = self._calculate_keyword_score(query, item["keywords"])
            
            # Combined score with smart weighting
            combined_score = (
                question_sim * 0.4 +      # Question similarity
                keyword_score * 0.5 +     # Keyword matching (highest weight)
                answer_sim * 0.1          # Answer relevance
            )
            
            scored_items.append({
                "item": item,
                "question_sim": question_sim,
                "keyword_score": keyword_score,
                "answer_sim": answer_sim,
                "combined_score": combined_score
            })
        
        # Sort by combined score
        scored_items.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top results
        results = []
        for scored in scored_items[:top_k]:
            results.append({
                "question": scored["item"]["question"],
                "answer": scored["item"]["answer"],
                "category": scored["item"]["category"],
                "question_similarity": scored["question_sim"],
                "keyword_score": scored["keyword_score"],
                "answer_similarity": scored["answer_sim"],
                "relevance_score": scored["combined_score"]
            })
        
        return results
    
    def _generate_response(self, query: str, context_items: List[Dict]) -> Tuple[str, float]:
        """Generate response using OpenAI with context"""
        
        if not context_items:
            return "I'm not sure about that. Want me to escalate it?", 0.3
        
        # Build comprehensive context
        context_text = "\n"
        for i, item in enumerate(context_items, 1):
            context_text += f"{i}. Question: {item['question']}\n"
            context_text += f"   Answer: {item['answer']}\n"
            context_text += f"   Relevance: {item['relevance_score']:.2f}\n"
            context_text += f"   Category: {item['category']}\n\n"
        
        print(f"🔍 Debug - Best relevance score: {max(item['relevance_score'] for item in context_items):.2f}")
        print(f"🔍 Debug - Using OpenAI with {len(context_items)} context items")
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{context_text}\nUser Question: {query}\n\nPlease provide a helpful answer based on the FAQ context above. Include specific details when available."}
                ],
                max_tokens=350,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"✅ OpenAI response received: {len(answer)} characters")
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, answer, context_items)
            print(f"🔍 Debug - Calculated confidence: {confidence:.2f}")
            
            return answer, confidence
            
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
            return "I'm having trouble processing your request right now. Want me to escalate it?", 0.2
    
    def _calculate_confidence(self, query: str, answer: str, context_items: List[Dict]) -> float:
        """Calculate confidence score with debugging"""
        
        # Check for escalation first
        if "I'm not sure" in answer or "Want me to escalate" in answer:
            print("🔍 Debug - Escalation phrase detected")
            return 0.25
        
        if not context_items:
            print("🔍 Debug - No context items")
            return 0.3
        
        # Get best relevance score
        best_relevance = max(item["relevance_score"] for item in context_items)
        print(f"🔍 Debug - Best relevance: {best_relevance:.2f}")
        
        # Keyword score from best match
        best_keyword_score = max(item["keyword_score"] for item in context_items)
        print(f"🔍 Debug - Best keyword score: {best_keyword_score:.2f}")
        
        # Answer quality indicators
        quality_indicators = [
            "$", "plan", "free", "trial", "support", "24/7", "security", 
            "encryption", "integration", "customize", "BlueBuck", "month",
            "features", "enterprise", "professional", "starter"
        ]
        
        answer_lower = answer.lower()
        quality_matches = sum(1 for indicator in quality_indicators if indicator.lower() in answer_lower)
        quality_score = min(0.2, quality_matches / len(quality_indicators))
        print(f"🔍 Debug - Quality matches: {quality_matches}/{len(quality_indicators)} = {quality_score:.2f}")
        
        # Answer length bonus (longer answers are usually better)
        length_bonus = min(0.1, len(answer) / 1500)
        print(f"🔍 Debug - Length bonus: {length_bonus:.2f}")
        
        # Calculate final confidence with FIXED weights
        confidence = (
            best_relevance * 0.3 +       # Relevance score (reduced weight)
            best_keyword_score * 0.4 +   # Keyword matching (increased weight) 
            quality_score * 0.2 +        # Answer quality
            length_bonus * 0.1 +         # Length bonus
            0.4                          # Higher base confidence
        )
        
        print(f"🔍 Debug - Base calculation: {confidence:.2f}")
        
        # Boost for high keyword scores (exact matches)
        if best_keyword_score > 0.3:  # Lower threshold
            confidence += 0.2  # Bigger boost
            print(f"🔍 Debug - Keyword boost applied: +0.2")
        
        # Special boost for good question similarity
        best_question_sim = max(item["question_similarity"] for item in context_items)
        if best_question_sim > 0.6:
            confidence += 0.15
            print(f"🔍 Debug - Question similarity boost: +0.15")
        
        # Cap confidence
        final_confidence = max(0.3, min(0.95, confidence))
        print(f"🔍 Debug - Final confidence: {final_confidence:.2f}")
        
        return final_confidence
    
    def query(self, question: str) -> Dict:
        """Process user query"""
        
        # Step 1: Smart retrieval
        context_items = self._smart_retrieval(question, top_k=3)
        
        # Step 2: Generate response
        answer, confidence = self._generate_response(question, context_items)
        
        # Step 3: Determine escalation
        escalate = confidence < self.confidence_threshold
        
        # Step 4: Build response
        response = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "escalate": escalate,
            "source": "simple_rag",
            "context_items": [
                {
                    "question": item["question"],
                    "category": item["category"],
                    "question_sim": item["question_similarity"],
                    "keyword_score": item["keyword_score"],
                    "relevance_score": item["relevance_score"]
                }
                for item in context_items
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(response)
        return response
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        total = len(self.conversation_history)
        escalated = sum(1 for r in self.conversation_history if r.get("escalate", False))
        
        return {
            "total_queries": total,
            "escalated_queries": escalated,
            "escalation_rate": escalated / total if total > 0 else 0,
            "confidence_threshold": self.confidence_threshold,
            "knowledge_base_size": len(self.faq_data),
            "avg_confidence": sum(r.get("confidence", 0) for r in self.conversation_history) / total if total > 0 else 0
        }

# Test the simple RAG agent
if __name__ == "__main__":
    print("🚀 Testing Simple Effective RAG Agent")
    print("=" * 55)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    
    try:
        # Initialize agent with optimized threshold
        agent = SimpleRAGAgent(openai_api_key=api_key, confidence_threshold=0.55)
        
        # Test queries
        test_queries = [
            "What is BlueBuck AI?",
            "How much does the Professional plan cost?",
            "Is there a free trial?",
            "What integrations do you support?",
            "How secure is my data?",
            "Do you have 24/7 support?",
            "Can I customize the AI models?",
            "What industries do you serve?",
            "How do I get started?",
            "Tell me about pricing",
            "What's the weather today?"  # Should escalate
        ]
        
        print("🧪 Testing Simple RAG System...")
        print("-" * 55)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}] ❓ Query: {query}")
            
            response = agent.query(query)
            
            print(f"🤖 Answer: {response['answer']}")
            print(f"📊 Confidence: {response['confidence']:.2f}")
            
            # Show scoring details
            if response['context_items']:
                print(f"📚 Top Match:")
                top_match = response['context_items'][0]
                print(f"   - {top_match['question'][:50]}...")
                print(f"   - Question Sim: {top_match['question_sim']:.2f}")
                print(f"   - Keyword Score: {top_match['keyword_score']:.2f}")
                print(f"   - Relevance: {top_match['relevance_score']:.2f}")
            
            if response['escalate']:
                print("⚠️  → ESCALATED")
            else:
                print("✅ → ANSWERED")
            
            print("-" * 35)
        
        # Show final stats
        print(f"\n📈 Final Statistics:")
        stats = agent.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Interactive mode
        print(f"\n🎯 Interactive Mode (type 'quit' to exit):")
        while True:
            try:
                user_input = input("\n💬 Ask about BlueBuck AI: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    response = agent.query(user_input)
                    print(f"\n🤖 {response['answer']}")
                    print(f"📊 Confidence: {response['confidence']:.2f}")
                    
                    # Show best match details
                    if response['context_items']:
                        best = response['context_items'][0]
                        print(f"🎯 Best Match: {best['question'][:40]}... (Relevance: {best['relevance_score']:.2f})")
                    
                    if response['escalate']:
                        print("⚠️  Query escalated")
                    else:
                        print("✅ Query answered successfully")
                    
            except KeyboardInterrupt:
                print("\n👋 Thanks for testing!")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
