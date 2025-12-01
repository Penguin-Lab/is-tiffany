import socket
from typing import Tuple, Union

from is_wire.core import Channel, Message


class StreamChannel(Channel):
    """Specialized class to consume only the latest message from a channel.

    Inherits from `is_wire.core.Channel` and adds the `consume_last` method to
    discard queued messages, returning only the most recent one. Ideal for
    real-time applications like video streaming, where processing outdated data
    is undesirable.
    """

    def __init__(
        self, uri: str = "amqp://guest:guest@10.10.2.211:30000", exchange: str = "is"
    ):
        """Initializes the streaming channel.

        Args:
            uri (str): URI of the AMQP server (broker).
            exchange (str): Name of the exchange to be used.
        """
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(
        self, return_dropped: bool = False
    ) -> Union[Message, Tuple[Message, int]]:
        """Consumes the latest available message from the channel, discarding previous ones.

        This method first waits for a message and then quickly consumes
        all subsequent messages already in the buffer, ensuring the returned
        message is the most recent.

        Args:
            return_dropped (bool): If True, also returns the number of messages
                                that were dropped. Defaults to False.

        Returns:
            msg (Message | Tuple[Message, int]]): The latest available message.
                                                    If `return_dropped` is True,
                                                    returns a tuple containing the
                                                    message and the number of dropped messages.

        Raises:
            socket.timeout: If no message is received within the timeout period
                            of the initial consume call.
        """
        dropped = 0
        msg = super().consume()
        while True:
            try:
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped) if return_dropped else msg
