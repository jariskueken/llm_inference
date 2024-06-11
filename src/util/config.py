from dataclasses import dataclass
import configparser


@dataclass
class ServerConfig:
    """
    Class to implement the config for the HTTP server that is running the LLM
    remotely
    """
    server: str
    model: str
    server_address: str
    server_port: int
    api_key: str

    def __init__(self,
                 server: str = "",
                 model: str = "",
                 server_address: str = "",
                 server_port: int = 0,
                 api_key: str = "",
                 ) -> None:
        """
        Create a config struct, can be either created manually or default
        values are used if config is created later using a config file

        :param server: The path to the server binary
        :param model: The path to the model file
        :param server_address: The host of the HTTP server
        :param server_port: The port of the HTTP server
        """
        self.server = server
        self.model = model
        self.server_address = server_address
        self.server_port = server_port
        self.api_key = api_key

    def build_config_from_file(self, f: str) -> None:
        config = configparser.ConfigParser()

        config.read(f)

        self.server_address = config.get("ServerConfig", "ServerAddress")
        self.server_port = config.get("ServerConfig", "ServerPort")
        self.server = config.get("ServerConfig", "Server")
        self.model = config.get("ServerConfig", "Model")
        self.api_key = config.get("ServerConfig", "ApiKey")
