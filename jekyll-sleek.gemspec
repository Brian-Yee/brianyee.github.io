# coding: utf-8

Gem::Specification.new do |spec|
  spec.name          = "Brian-Yee"
  spec.version       = "0.1.0"
  spec.authors       = ["Brian Yee"]
  spec.email         = ["brian.ph.yee@gmail.com"]

  spec.summary       = %q{Toronto based ML Consultant and Data Scientist}
  spec.homepage      = "http://brianyee.ai"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r!^(assets|_(includes|layouts|sass)/|(LICENSE|README)((\.(txt|md|markdown)|$)))!i)
  end

  spec.platform      = Gem::Platform::RUBY
  spec.add_runtime_dependency "jekyll", "~> 3.6"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.3"
  spec.add_runtime_dependency "jekyll-sitemap", "~> 1.1"

  spec.add_development_dependency "bundler", "~> 1.12"
  spec.add_development_dependency "rake", "~> 10.0"
end
