# National Real Estate Market Data — United States

## Median Home Prices by Major Metro Area (2024 Estimates)

### West Coast
- **San Francisco, CA:** Median $1,350,000 | Avg $750/sqft | High demand, tech-driven
- **San Jose, CA:** Median $1,400,000 | Avg $780/sqft | Silicon Valley premium
- **Los Angeles, CA:** Median $950,000 | Avg $580/sqft | Entertainment, diverse economy
- **San Diego, CA:** Median $875,000 | Avg $520/sqft | Military, biotech, tourism
- **Seattle, WA:** Median $780,000 | Avg $450/sqft | Tech hub, Amazon/Microsoft
- **Portland, OR:** Median $520,000 | Avg $310/sqft | Growing tech scene
- **Sacramento, CA:** Median $510,000 | Avg $320/sqft | State capital, Bay Area spillover

### Mountain West
- **Denver, CO:** Median $580,000 | Avg $340/sqft | Outdoor lifestyle, growing tech
- **Salt Lake City, UT:** Median $520,000 | Avg $290/sqft | Tech growth, outdoor recreation
- **Phoenix, AZ:** Median $430,000 | Avg $270/sqft | Retirement, rapid growth
- **Las Vegas, NV:** Median $410,000 | Avg $240/sqft | Tourism, entertainment economy
- **Boise, ID:** Median $450,000 | Avg $260/sqft | Remote work migration
- **Tucson, AZ:** Median $310,000 | Avg $200/sqft | University town, retirement

### Midwest
- **Chicago, IL:** Median $320,000 | Avg $220/sqft | Finance, diverse economy
- **Minneapolis, MN:** Median $340,000 | Avg $210/sqft | Healthcare, corporate HQs
- **Columbus, OH:** Median $280,000 | Avg $180/sqft | University, insurance, tech growth
- **Indianapolis, IN:** Median $260,000 | Avg $160/sqft | Affordable, logistics hub
- **Kansas City, MO:** Median $270,000 | Avg $165/sqft | Affordable, growing tech
- **Detroit, MI:** Median $220,000 | Avg $130/sqft | Automotive, revitalizing downtown
- **Cleveland, OH:** Median $200,000 | Avg $120/sqft | Healthcare, affordable
- **St. Louis, MO:** Median $230,000 | Avg $140/sqft | Healthcare, education

### South
- **Austin, TX:** Median $550,000 | Avg $320/sqft | Tech hub, rapid growth
- **Dallas, TX:** Median $380,000 | Avg $220/sqft | Corporate relocations, diverse economy
- **Houston, TX:** Median $330,000 | Avg $180/sqft | Energy, medical center
- **San Antonio, TX:** Median $290,000 | Avg $170/sqft | Military, tourism, affordable
- **Nashville, TN:** Median $430,000 | Avg $280/sqft | Entertainment, healthcare, growth
- **Charlotte, NC:** Median $380,000 | Avg $220/sqft | Banking, growing metro
- **Raleigh, NC:** Median $420,000 | Avg $240/sqft | Research triangle, tech
- **Atlanta, GA:** Median $370,000 | Avg $210/sqft | Logistics, entertainment, diverse
- **Tampa, FL:** Median $380,000 | Avg $250/sqft | Growth, retirement, tourism
- **Miami, FL:** Median $580,000 | Avg $400/sqft | International finance, luxury market
- **Orlando, FL:** Median $370,000 | Avg $230/sqft | Tourism, theme parks, growth
- **Jacksonville, FL:** Median $310,000 | Avg $190/sqft | Military, logistics, affordable

### Northeast
- **New York City, NY:** Median $750,000 | Avg $650/sqft | Global finance, extremely dense
- **Boston, MA:** Median $720,000 | Avg $480/sqft | Education, biotech, healthcare
- **Washington, DC:** Median $620,000 | Avg $380/sqft | Government, defense, consulting
- **Philadelphia, PA:** Median $320,000 | Avg $200/sqft | Healthcare, education, affordable
- **Pittsburgh, PA:** Median $230,000 | Avg $150/sqft | Tech growth, healthcare, affordable
- **Baltimore, MD:** Median $280,000 | Avg $170/sqft | Healthcare, government adjacent
- **Hartford, CT:** Median $300,000 | Avg $190/sqft | Insurance, suburban

## Price-Per-Square-Foot Benchmarks by Region

Understanding $/sqft norms is essential for cross-market comparisons:

- **Ultra-High Cost (> $500/sqft):** San Francisco, San Jose, New York City, Manhattan
- **High Cost ($350-500/sqft):** Los Angeles, Seattle, Boston, Miami, Washington DC
- **Above Average ($250-350/sqft):** Denver, Austin, San Diego, Nashville, Portland, Raleigh
- **Average ($180-250/sqft):** Dallas, Chicago, Charlotte, Atlanta, Tampa, Phoenix, Orlando
- **Affordable ($120-180/sqft):** Houston, Indianapolis, Columbus, San Antonio, Jacksonville
- **Budget (< $120/sqft):** Detroit, Cleveland, Memphis, certain rural markets

## Key Price Drivers (Universal)

These factors affect property prices in every US market:

### Location Factors
- **School district quality:** Top-rated districts command 10-30% premium
- **Crime rates:** Low-crime neighborhoods see 5-15% higher values
- **Proximity to employment centers:** Each mile further reduces value 1-3%
- **Public transit access:** Properties near stations see 5-20% premium
- **Walkability score:** Walk Score 70+ adds 5-15% to urban property values
- **Natural amenities:** Water views, mountain views, parks add 10-50% premium

### Property Factors
- **Square footage:** Most fundamental driver. Larger homes cost more but diminishing returns above 3000sqft
- **Bedrooms/Bathrooms:** 3BR/2BA is the most marketable configuration in most markets
- **Age and condition:** New construction commands 15-25% premium over similar older homes
- **Lot size:** Larger lots add value in suburban/rural areas, less impact in urban
- **Garage:** 2-car garage adds $20K-$50K depending on market
- **Updates/Renovations:** Kitchen and bathroom updates yield highest ROI (60-80% return)

### Market Factors
- **Interest rates:** Each 1% rate increase reduces buying power by ~10%
- **Local job growth:** Markets with strong job growth see 5-10% annual appreciation
- **Supply vs demand:** Low inventory markets see faster appreciation
- **Seasonal patterns:** Spring/summer listings typically sell 5-10% higher than winter

## How to Estimate Property Price Without a Trained Model

When an ML model is not available for a specific market, use this systematic approach:

1. **Identify the metro area** and its median home price
2. **Calculate base price** using median $/sqft for the area multiplied by property sqft
3. **Adjust for bedrooms/bathrooms:** More rooms than market average adds 3-5% per extra room
4. **Adjust for condition:** Poor condition -10 to -20%, excellent condition +5 to +15%
5. **Adjust for age:** Newer homes +5 to +15%, very old homes -5 to -15%
6. **Adjust for special features:** Waterfront +30-100%, good views +10-25%, pool +5-10%
7. **Apply neighborhood factor:** Premium neighborhood +10-30%, budget neighborhood -10-20%
8. **Calculate confidence range:** Typically +/- 15-25% for LLM-based estimates

This approach provides a reasonable estimate but should always be validated with local market data and professional appraisals.

## Regional Market Trends

### High-Growth Markets (2023-2024)
Markets seeing above-average appreciation: Austin, Nashville, Raleigh, Boise, Phoenix, Tampa, Charlotte. Driven by remote work migration, corporate relocations, and quality-of-life factors.

### Stable Markets
Markets with steady, moderate growth: Seattle, Denver, Dallas, Chicago, Atlanta, Washington DC. Mature markets with diverse economic bases.

### Recovery Markets
Markets rebuilding after corrections: San Francisco (post-tech adjustment), New York City (post-pandemic urban return), Detroit (long-term revitalization).

## Important Disclaimer

All price estimates for locations outside King County, WA are based on general market data and LLM reasoning, NOT trained machine learning models. These estimates:
- Have wider confidence intervals (typically +/- 15-25%)
- Should be treated as rough approximations
- Must be verified with local real estate professionals
- Do not account for micro-neighborhood variations
- May not reflect the most current market conditions
