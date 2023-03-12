# CI

This continuos integration material is from university course.  
This project unit test, integrations tests and end-to-end tests  
a phonebook app I coded. I will concentrate on backend, tests and CI.

App used MongoDb as database. Basic entity person is as follows.

```javascript
const personSchema = new mongoose.Schema({
  name: {
    type: String,
    minlength: 3,
    required: true,
  },
  number: {
    type: String,
    minlength: 8,
    required: true,
    validate: {
      validator: (n) => {
        return /\d{2}-\d+/.test(n) || /\d{3}-\d+/.test(n);
      },
      message: (props) => `${props.value} is not a valid phone number`,
    },
  },
});
```

APi calls for fetching the person are pretty simple rest calls. All  
rest protocol queries were cover, but here's get.

```javascript
app.get("/api/persons/:id", (req, res, next) => {
  Person.findById(req.params.id)
    .then((person) => {
      res.json(person);
    })
    .catch((error) => next(error));
});
```

Backend test were simple. Here is example of one test  
for the backend.

```javascript
test("invalid contact is not added", async () => {
  const newContact = {
    name: "john doe",
    number: "10021123",
  };
  await api.post("/api/persons").send(newContact).expect(400);
});
```

Now to the actual CI part. Project had CI done one github actions.  
Workflow file was pretty lengthy. It is at the bottom if you're  
interested. In short workflow deployed to heroku. Test were done,  
env set up and notifications of test failures set up, all by making  
Pull-Request from development branch into master branch.

```yml
name: Deployment pipeline
on:
  pull_request:
    branches: [master]
    types: [opened, synchronize]

jobs:
  build_and_deploy:
    runs-on: ubuntu-20.04
    steps:
      - uses: actions/checkout@v3

      - uses: mstachniuk/ci-skip@v1
        with:
          fail-fast: true
          commit-filter: "#skip"

      - uses: actions/setup-node@v2
        with:
          node-version: "16"

      - uses: SpicyPizza/create-envfile@v1.3
        with:
          envkey_MONGODB_URI: ${{ secrets.MONGODB_URI }}
          envkey_PORT: ${{ secrets.PORT }}
          file_name: .env
          fail_on_empty: true

      - name: npm install backend
        run: npm install
      - name: npm install frontend
        working-directory: ./frontend
        run: npm install
      - name: lint
        run: npm run lint
      - name: build
        run: npm run build:ui
      - name: test backend
        run: npm run test
      - name: test frontend
        working-directory: ./frontend
        run: npm run test

      - name: tests e2e
        uses: cypress-io/github-action@v2
        with:
          command: npm run test:e2e
          start: npm run start
          wait-on: http://localhost:3001

      - name: deploy
        uses: akhileshns/heroku-deploy@v3.12.12 # This is the action
        with:
          heroku_api_key: ${{secrets.HEROKU_API_KEY}}
          heroku_app_name: "shielded-shore-74120" #Must be unique in Heroku
          heroku_email: "kaimhall@gmail.com"
          healthcheck: "https://shielded-shore-74120.herokuapp.com/health"
          checkstring: "ok"
          rollbackonhealthcheckfailed: true

  tag_release:
    needs: build_and_deploy
    runs-on: ubuntu-20.04
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: "0"
      - name: Bump version and push tag
        uses: anothrNick/github-tag-action@master
        env:
          pre_release: false
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DEFAULT_BUMP: patch

  discord_notify:
    runs-on: ubuntu-20.04
    needs: tag_release
    steps:
      - name: test success
        uses: rjstone/discord-webhook-notify@v1.0.4
        if: ${{ success() }}
        with:
          severity: info
          description: Test Succeeded!
          webhookUrl: ${{ secrets.DISCORD_WEBHOOK }}
          username: kaimhall
      - name: test failure
        uses: rjstone/discord-webhook-notify@v1.0.4
        if: ${{ failure() }}
        with:
          severity: error
          webhookUrl: ${{ secrets.DISCORD_WEBHOOK }}
          username: kaimhall
```

[home](https://kaimhall.github.io/portfolio/)
