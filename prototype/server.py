# based on https://www.polarsparc.com Primer on MCP/LangChain/ReAct

import logging
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("interest_mcp_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s - %(message)s"
)

mcp = FastMCP("InterestCalculator")

@mcp.tool()
def yearly_simple_interest(principal: float, rate: float) -> float:
    """
    Calculate yearly simple interest based on principal and interest rate.

    Args:
        principal: The initial amount of money invested or loaned (e.g. 1000.0).
        rate: Annual interest rate as a percentage (e.g. 5.0 for 5%).

    Returns:
        The simple interest earned after one year.
    """
    logger.info("Calculating simple interest: Principal=%.2f, Rate=%.2f", principal, rate)
    return principal * rate / 100.0


@mcp.tool()
def yearly_compound_interest(principal: float, rate: float) -> float:
    """
    Calculate yearly compound interest based on principal and interest rate.

    Args:
        principal: The initial amount of money invested or loaned (e.g. 1000.0).
        rate: Annual interest rate as a percentage (e.g. 5.0 for 5%).

    Returns:
        The amount after one year with interest compounded annually.
    """
    logger.info("Calculating compound interest: Principal=%.2f, Rate=%.2f", principal, rate)
    return principal * (1 + rate / 100.0)

def main():
    logger.info("Starting the Interest MCP Server...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
