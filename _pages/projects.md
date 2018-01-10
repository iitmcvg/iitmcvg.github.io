---
title: "Projects"
layout: single
permalink: "/projects/"
author_profile: true
---
## Completed Projects
{% for item in site.completed-projects %}
  <h2><a href="{{ item.url }}">{{ item.title }}</a></h2>
  <p>{{ item.description }}</p>
{% endfor %}

## Ongoing Projects
{% for item in site.ongoing-projects %}
  <h2><a href="{{ item.url }}">{{ item.title }}</a></h2>
  <p>{{ item.description }}</p>
{% endfor %}
