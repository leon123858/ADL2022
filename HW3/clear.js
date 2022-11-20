const fs = require('fs-extra');

async function remove(dest) {
	try {
		await fs.remove(dest);
	} catch (err) {
		console.error(err);
	}
}

(async () => {
	await remove('./report/data');
	await remove('./report/cache');
	await remove('./report/model');
	await remove('./report/libs');
	await remove('./report/src');
	await remove('./report/colab');
	await remove('./report/data.zip');
	await remove('./report/model.zip');
})();
