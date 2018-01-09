---
title: "Completed Projects"
layout: single
permalink: "/completed-projects/"
author_profile: true
---

{% for item in site.documentation %}
  <h2><a href="{{ item.url }}">{{ item.title }}</a></h2>
  <p>{{ item.description }}</p>
{% endfor %}
