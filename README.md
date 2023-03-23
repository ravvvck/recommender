# Book recommender

A book recommendation application that supports both content-based and collaborative filtering recommendations. 
For content-based recommendations, it uses the tf-idf algorithm for the Genres column in the book collection. To get a recommendation of similar books, you need to send the book id.

Collaborative filtering uses the [Implicit](https://implicit.readthedocs.io/en/latest/) library. Recommendations are created based on sent user ratings in the following format:
```json
[{
  "user_id": 1,
  "rating": 5,
  "book_id": 27005
},
{
  "user_id": 1,
  "rating": 4,
  "book_id": 12768
}]
```
Ratings are integers between 1-5.
