const fs = require('fs-extra');

async function copy(src, dst) {
	try {
		await fs.copy(src, dst);
	} catch (err) {
		console.error(err);
	}
}

(async () => {
	await copy('./libs', './report/libs');
	await copy('./src', './report/src');
})();
