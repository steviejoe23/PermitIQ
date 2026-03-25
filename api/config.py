from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "dev"              # dev | prod
    STRICT_SCHEMA: bool = False   # True in prod
    ENABLE_DOCS: bool = True

    SCHEMA_NAME: str = "boston_parcel_zoning"
    SCHEMA_VERSION: str = "1.0.0"

settings = Settings()
