import os
import asyncio
import psycopg2
import pinecone
import logging
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="embedding"
)

async def get_summaries_to_embed() -> List[Dict[str, Any]]:
    """Get conversation summaries that need to be embedded."""
    conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, username, summary, timestamp
            FROM conversation_summaries
            WHERE embedded_at IS NULL
            ORDER BY timestamp ASC
            LIMIT 50
            """
        )
        
        summaries = []
        for row in cur.fetchall():
            summaries.append({
                "id": row[0],
                "username": row[1],
                "summary": row[2],
                "timestamp": row[3]
            })
        
        return summaries
    except Exception as e:
        logger.error(f"Error getting summaries: {e}")
        raise
    finally:
        cur.close()
        conn.close()

async def create_embedding(text: str) -> List[float]:
    """Create embedding for text using OpenAI API."""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise

async def store_embedding_in_pinecone(
    summary_id: int, 
    username: str, 
    summary: str, 
    embedding: List[float]
):
    """Store embedding in Pinecone."""
    try:
        # Get Pinecone index
        index = pinecone.Index("embedding")
        
        # Create metadata
        metadata = {
            "username": username,
            "content": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        # Upsert embedding
        index.upsert(
            vectors=[
                {
                    "id": f"summary_{summary_id}",
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        logger.info(f"Stored embedding for summary {summary_id} in Pinecone")
    except Exception as e:
        logger.error(f"Error storing embedding in Pinecone: {e}")
        raise

async def mark_summary_as_embedded(summary_id: int):
    """Mark summary as embedded in database."""
    conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE conversation_summaries
            SET embedded_at = %s
            WHERE id = %s
            """,
            (datetime.now(), summary_id)
        )
        conn.commit()
        logger.info(f"Marked summary {summary_id} as embedded")
    except Exception as e:
        logger.error(f"Error marking summary as embedded: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

async def process_summaries():
    """Process all summaries that need embedding."""
    try:
        summaries = await get_summaries_to_embed()
        logger.info(f"Found {len(summaries)} summaries to embed")
        
        for summary in summaries:
            try:
                # Create embedding
                embedding = await create_embedding(summary["summary"])
                
                # Store in Pinecone
                await store_embedding_in_pinecone(
                    summary["id"],
                    summary["username"],
                    summary["summary"],
                    embedding
                )
                
                # Mark as embedded
                await mark_summary_as_embedded(summary["id"])
                
                # Sleep briefly to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing summary {summary['id']}: {e}")
                continue
    except Exception as e:
        logger.error(f"Error in process_summaries: {e}")

async def main():
    """Main function to run the embedding process."""
    try:
        # Check environment variables
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "POSTGRES_CONNECTION_STRING"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
        
        logger.info("Starting embedding process...")
        await process_summaries()
        logger.info("Embedding process completed")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 