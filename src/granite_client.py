"""
IBM Granite LLM Integration via IBM watsonx.ai
"""
import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, EmbeddingTypes

load_dotenv()


class GraniteClient:
    """Wrapper for IBM Granite model via watsonx.ai"""

    def __init__(self):
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.model_id = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-3-8b-instruct")

        if not self.api_key or not self.project_id:
            raise ValueError(
                "WATSONX_API_KEY and WATSONX_PROJECT_ID must be set in .env file"
            )

        self.credentials = Credentials(
            url=self.url,
            api_key=self.api_key,
        )
        self.client = APIClient(self.credentials)
        self._model = None

    @property
    def model(self) -> ModelInference:
        """Lazy-initialize the Granite model."""
        if self._model is None:
            self._model = ModelInference(
                model_id=self.model_id,
                credentials=self.credentials,
                project_id=self.project_id,
            )
        return self._model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: list = None,
    ) -> str:
        """Generate text using IBM Granite model."""
        params = {
            GenParams.MAX_NEW_TOKENS: max_tokens,
            GenParams.TEMPERATURE: temperature,
            GenParams.TOP_P: top_p,
            GenParams.REPETITION_PENALTY: 1.1,
        }
        if stop_sequences:
            params[GenParams.STOP_SEQUENCES] = stop_sequences

        response = self.model.generate_text(
            prompt=prompt,
            params=params,
        )
        return response.strip()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using IBM slate embedding model."""
        embedding_model_id = os.getenv(
            "GRANITE_EMBEDDING_MODEL_ID", "ibm/slate-125m-english-rtrvr"
        )
        from ibm_watsonx_ai.foundation_models import Embeddings

        embedding_model = Embeddings(
            model_id=embedding_model_id,
            credentials=self.credentials,
            project_id=self.project_id,
        )
        response = embedding_model.embed_documents(texts=texts)
        return response


# Singleton instance
_granite_client = None


def get_granite_client() -> GraniteClient:
    """Get or create the singleton GraniteClient."""
    global _granite_client
    if _granite_client is None:
        _granite_client = GraniteClient()
    return _granite_client
