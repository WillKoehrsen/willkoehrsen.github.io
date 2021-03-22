# Notes for blogging

### Center text below single images
- Edit styles.scss to include img + em
- *image caption* on line directly below image

### Display multiple images
- Create a table
- Currently only works for 1 x 2
- Must include header in table


UMAP | PCA
:---:|:---:
![](https://cdn-images-1.medium.com/max/1200/1*F4F_vnQXiB5RjGNZUOWwug.png) |  ![](https://cdn-images-1.medium.com/max/1200/1*pkQg_N4T-ersZ86ePFt20g.png)


### Include GitHub gist
- Go to article on medium, right click on gist, select inspect frame
- Copy and paste the script tags

<script src="https://gist.github.com/WillKoehrsen/27d12fba73d729cd3c50b20442925087.js" charset="utf-8">
</script>
<center>Recursive Feature Elimination code.</center>

### Center text below code blocks
<center>Centered Text</center>

### Code highlighting with rouge
- Install rouge
	* gem install rouge
- Generate a style sheet
	* rougify style "style" >> sass/code-highlighting.scss"
- Change styles.scss to include @import "code-highlighting"
- View available rouge styles at https://spsarolkar.github.io/rouge-theme-preview/
- Approach taken from https://bnhr.xyz/2017/03/25/add-syntax-highlighting-to-your-jekyll-site-with-rouge.html

### Include live code from repl.it
- Include repl link with ?lite=true at end of url
- Place inside iframe inside div

<div class="code-container">
    <iframe src="https://repl.it/@WillKoehrsen/basicpython?lite=true">
    </iframe>
</div>

## Replacing Code Tags

### Starting
Find: <pre(.*?)>
Replace: ```\n

#### Ending

Find: </pre(.*?)>
Replace: \n```