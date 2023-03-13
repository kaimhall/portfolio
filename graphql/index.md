These snippets are from fullstack open Grapghql. This app was about  
books and authors. there was async api calls etc.

App was pretty standard react app. It used context for state.  
I try demonstrate graphql query language in this portion.

App used apollo server.

```javascript
import { gql } from "@apollo/client";

export const ALL_BOOKS = gql`
  query allBooks($author: String, $genre: String) {
    allBooks(author: $author, genre: $genre) {
      title
      author {
        name
        id
        born
        bookCount
      }
      published
      genres
    }
  }
`;
```

Graphql has nice nested object form in its queries.  
Above query gets author by genre from all books. It's  
response carries some author details.

```javascript
export const EDIT_AUTHOR = gql`
  mutation editAuthor($name: String!, $setBornTo: Int!) {
    editAuthor(name: $name, setBornTo: $setBornTo) {
      name
      id
      born
      bookCount
    }
  }
`;
```

editing is called mutations in graphql. Dollar sign are params  
like you probably noticed.

[home](https://kaimhall.github.io/portfolio/)
