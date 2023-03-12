# Sql

This material presents first few router used in routing  
SQL queries. Below router sends get request for certain  
userid address. User is then searched by primarykey.  
Blogs (object) written by user are in response also.

```sql
router.get('/:id', async (req, res) => {
  let where = {}

  if (req.query.read) {
    where.read = req.query.read === 'true'
  }

  const user = await User.findByPk(req.params.id, {
    attributes: ['name', 'username'],
    include: [
      {
        model: Blog,
        as: 'readings',
        attributes: { exclude: ['createdAt', 'updatedAt', 'userId'] },
        through: {
          attributes: ['id', 'read'],
          where,
        },
      },
    ],
  })
  if (user) {
    res.send(user)
  } else {
    res.status(404).send({ error: 'user blog not found' })
  }
})
```

User class (model) has id, username and name fields. Below is  
postgreql veraion of modeling that.

```sql
class User extends Model {}

User.init(
  {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true,
    },
    username: {
      type: DataTypes.STRING,
      unique: true,
      allowNull: false,
      validate: {
        isEmail: {
          args: true,
          msg: 'username must be an email',
        },
      },
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    disabled: {
      type: DataTypes.BOOLEAN,
      defaultValue: false,
    },
  },
  {
    sequelize,
    underscored: true,
    timestamps: true,
    modelName: 'user',
  }
)
```
