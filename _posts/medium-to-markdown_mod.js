const mediumToMarkdown = require('medium-to-markdown');
 
mediumToMarkdown.convertFromUrl("https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815").then(function (markdown) {
  console.log(markdown); //=> Markdown content of medium post
});
