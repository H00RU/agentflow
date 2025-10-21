"""
WebSearch Operator for AIME Mathematical Problem Solving
Allows workflows to search for mathematical knowledge and formulas
"""

import asyncio
from typing import Dict, Any, Optional
import aiohttp
from scripts.async_llm import AsyncLLM


class WebSearchOperator:
    """
    WebSearch operator that allows searching for mathematical knowledge

    This operator uses DuckDuckGo or similar search APIs to find
    relevant mathematical formulas, theorems, and solution methods.
    """

    def __init__(self, llm: AsyncLLM):
        """
        Args:
            llm: LLM instance for processing search results
        """
        self.llm = llm

    async def __call__(
        self,
        query: str,
        problem_context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform web search for mathematical knowledge

        Args:
            query: Search query
            problem_context: Optional problem context for better search

        Returns:
            Dict containing search results and synthesized knowledge
        """
        try:
            # Construct search query
            if problem_context:
                search_query = f"{query} mathematics {problem_context}"
            else:
                search_query = f"{query} mathematics formula theorem"

            # Perform search (using DuckDuckGo HTML search)
            search_results = await self._perform_search(search_query)

            # Synthesize results using LLM
            synthesis_prompt = f"""Based on these search results about "{query}", provide a concise summary of relevant mathematical knowledge:

Search Results:
{search_results}

Provide:
1. Key formulas or theorems
2. Solution approaches
3. Important considerations

Keep it concise and focused on what's useful for solving the problem."""

            synthesis = await self.llm.call(
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3
            )

            return {
                'response': synthesis,
                'query': query,
                'raw_results': search_results[:500]  # Truncate for efficiency
            }

        except Exception as e:
            return {
                'response': f"Search failed: {str(e)}. Proceeding without external knowledge.",
                'query': query,
                'raw_results': ""
            }

    async def _perform_search(self, query: str, max_results: int = 5) -> str:
        """
        Perform web search using DuckDuckGo HTML

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            Formatted search results as string
        """
        try:
            # Use DuckDuckGo HTML search (no API key needed)
            url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'kl': 'us-en'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()
                        results = self._parse_duckduckgo_html(html, max_results)
                        return results
                    else:
                        return f"Search returned status {response.status}"

        except asyncio.TimeoutError:
            return "Search timeout - using general mathematical knowledge"
        except Exception as e:
            return f"Search error: {str(e)}"

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> str:
        """
        Parse DuckDuckGo HTML results

        Args:
            html: Raw HTML response
            max_results: Maximum results to extract

        Returns:
            Formatted results string
        """
        try:
            # Simple extraction of result snippets
            results = []

            # Find result containers
            import re
            snippets = re.findall(r'class="result__snippet">(.*?)</a>', html, re.DOTALL)

            for i, snippet in enumerate(snippets[:max_results]):
                # Clean HTML tags
                clean_snippet = re.sub(r'<[^>]+>', '', snippet)
                clean_snippet = clean_snippet.strip()
                if clean_snippet:
                    results.append(f"{i+1}. {clean_snippet[:200]}")

            if results:
                return "\n\n".join(results)
            else:
                return "No specific results found. Using general mathematical knowledge."

        except Exception as e:
            return f"Parsing error: {str(e)}"


# Operator registration name
operator_name = "WebSearch"
operator_class = WebSearchOperator
