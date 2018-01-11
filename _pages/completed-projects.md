---
title: "Completed Projects"
layout: single
permalink: "/completed-projects/"
author_profile: true
---

{% for item in site.completed-projects %}
  <h3><a href="{{ item.url }}">{{ item.title }}</a></h3>
  <p>{{ item.description }}</p>
{% endfor %}
