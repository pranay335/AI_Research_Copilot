from collections import defaultdict


def extract_trends(articles, topics_per_article):
    """
    Analyze how topics evolve over time based on article dates.
    articles: list of dicts with 'id' and 'date'
    topics_per_article: dict mapping article_id to a list of keywords
    """
    trends = defaultdict(lambda: defaultdict(int))
    
    for article in articles:
        date = article.get("date")
        article_id = article.get("id")
        
        if not date or not article_id:
            continue
            
        period = date[:7] # YYYY-MM
        
        topics = topics_per_article.get(article_id, [])
        for topic in topics:
            trends[period][topic] += 1
            
    formatted_trends = {}
    for period in sorted(trends.keys()):
        top_period_topics = sorted(
            trends[period].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        formatted_trends[period] = {
            "topics": [{"topic": k, "mentions": v} for k, v in top_period_topics]
        }
        
    return formatted_trends
