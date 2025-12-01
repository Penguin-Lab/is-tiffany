import re

from is_wire.core import AsyncTransport, Logger
from is_wire.rpc import LogInterceptor, ServiceProvider, TracingInterceptor
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter

from .StreamChannel import StreamChannel


class Connection:
    """
    Manages connections to the message broker and Zipkin tracing server.

    This class encapsulates the logic for establishing communication channels,
    providing RPC services, subscribing to topics (e.g., camera frames),
    and setting up distributed tracing with Zipkin.

    Attributes:
        channel (StreamChannel): Channel for the main service provider.
        provider (ServiceProvider): Manages and exposes RPC services.
        exporter (ZipkinExporter): Exporter to send traces to Zipkin.
        broker_uri (str): URI of the message broker.
        zipkin_uri (str): URI of the Zipkin server.
        service_name (str): Name of this service.
    """

    def __init__(self, broker_uri: str, zipkin_uri: str, service_name: str) -> None:
        self.channel = StreamChannel(broker_uri)
        self.provider = ServiceProvider(self.channel)

        log = LogInterceptor()
        self.provider.add_interceptor(log)
        self.log = log.log

        self.log.info(f"Successfully connected to broker at {broker_uri}")

        self.exporter = self.create_exporter(service_name, zipkin_uri, log.log)
        self.provider.add_interceptor(TracingInterceptor(self.exporter))
        self.log.info(
            f"Zipkin exporter initialized for service '{service_name}' with URI: {zipkin_uri}"
        )

        self.broker_uri = broker_uri
        self.zipkin_uri = zipkin_uri
        self.service_name = service_name

    @staticmethod
    def create_exporter(service_name: str, uri: str, log: Logger) -> ZipkinExporter:
        """
        Creates and configures a ZipkinExporter for distributed tracing.

        This method validates the Zipkin URI format and initializes an exporter
        with the provided service name and an asynchronous transport layer.

        Args:
            service_name (str): Name of the service to be shown in Zipkin.
            uri (str): URI of the Zipkin server, expected format 'http://<hostname>:<port>'.
            log (Logger): Logger instance to record messages, especially errors.

        Returns:
            ZipkinExporter: Configured exporter ready to send traces.

        Raises:
            ValueError: If the provided URI does not match the expected format.
        """
        zipkin_ok = re.match(r"http://([a-zA-Z0-9\.-]+)(:(\d+))?", uri)
        if not zipkin_ok:
            log.critical(
                'Invalid Zipkin URI "{}", expected http://<hostname>:<port>', uri
            )

        exporter = ZipkinExporter(
            service_name=service_name,
            host_name=zipkin_ok.group(1),
            port=zipkin_ok.group(3),
            transport=AsyncTransport,
        )
        return exporter
