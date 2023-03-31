import typing
import strawberry
from fastapi import FastAPI
from strawberry.asgi import GraphQL

#@strawberry.input
#class SearchInput:
#    search_string: str

       
@strawberry.type
class SearchResult:
    score: float
    title: str
    snippet: str
    link: str

def search(self, search_input: str) -> SearchResult:
    if search_input == "Getting Started":
        return SearchResult(score=1.0, title="Getting Started", snippet="hi snippet data", link="/doc/getting-started.md")
    else:
        return SearchResult(score=1.0, title="Apple Pay", snippet="apple pay", link="/doc/apple-pay.md")
    
@strawberry.type
class Query:
    results: SearchResult = strawberry.field(resolver=search)

    

schema = strawberry.Schema(query=Query)


graphql_app = GraphQL(schema)

app = FastAPI()
app.add_route("/graphql", graphql_app)
app.add_websocket_route("/graphql", graphql_app)